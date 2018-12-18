using Distributed
# addprocs(2)
@everywhere include("model.jl")
include("model.jl")

import Random
using Dates: now

using HDF5
using PyCall
import JSON
using LatinHypercubeSampling
using Parameters

@pyimport numpy
@pyimport skopt
# Optimizer methods
ask(opt)::Vector{Float64} = opt[:ask]()
tell(opt, x::Vector{Float64}, fx::Float64) = opt[:tell](Tuple(x), fx)


# Parameters.
@with_kw struct Job
    n_arm::Int = 2
    base_obs_sigma::Float64 = 1
    diff_obs_sigma::Float64 = 1
    sample_cost::Float64 = 0.001
    switch_cost::Float64 = 1

    n_iter::Int = 100
    n_roll::Int = 1000
    n_sim::Int = 10000
    seed::Int = 0
    group::String = "dummy"
end
function Job(file::AbstractString)
    kws = Dict(Symbol(k)=>v for (k, v) in JSON.parsefile(file))
    Job(;kws...)
end
Params(job::Job) = Params(
    job.n_arm,
    job.base_obs_sigma,
    job.diff_obs_sigma,
    job.sample_cost,
    job.switch_cost,
)
Base.string(job::Job) = join((getfield(job, f) for f in fieldnames(Job)), "-")
result_file(job::Job, name) = "runs/$(job.group)/results/$name-$(string(job)).json"

function save(job::Job, name::Symbol, value)
    d = Dict(
        :job => job,
        :time => now(),
        name => value
    )
    open(result_file(job, name), "w") do f
        write(f, JSON.json(d))
    end
    println("Wrote $(result_file(job, name))")
end
load(job::Job, name::Symbol) = JSON.parsefile(result_file(job, name))[string(name)]

# Optimization.
function x2theta(x)
    cost_weight = x[1]
    voi_weights = diff([0; sort(collect(x[2:end])); 1])
    [cost_weight; voi_weights]
end

function max_cost(prm::Params)
    theta = Float64[1, 0, 0, 1]
    s = State(prm)
    b = Belief(s)
    computes() = Policy(prm, theta)(b) != TERM

    while computes()
        theta[1] *= 2
    end

    while !computes()
        theta[1] /= 2
        if theta[1] < 2^-10
            error("Computation is too expensive")
        end
    end

    step_size = theta[1] / 100
    while computes()
        theta[1] += step_size
    end
    theta[1]
end



function optimize(job::Job)
    @unpack seed, n_iter, n_roll = job
    Random.seed!(seed)
    numpy.random[:seed](seed)
    prm = Params(job)

    function loss(x; nr=n_roll)
        policy = Policy(prm, x2theta(x))
        reward = @distributed (+) for i in 1:nr
            rollout(prm, policy, max_steps=200).reward
        end
        - reward
    end
    bounds = [ (0., max_cost(prm)), (0., 1.), (0., 1.) ]
    opt = skopt.Optimizer(bounds, random_state=seed)


    # Choose initial samples by Latin Hypersquare sampling.
    upper_bounds = [b[2] for b in bounds]
    n_latin = max(2, cld(n_iter, 4))
    latin_points = LHCoptim(n_latin, length(bounds), 1000)[1]
    for i in 1:n_latin
        x = latin_points[i, :] ./ n_latin .* upper_bounds
        tell(opt, x, loss(x))
    end

    # Bayesian optimization.
    for i in 1:(n_iter - n_latin)
        x = ask(opt)
        tell(opt, x, loss(x))
    end

    # Cross validation.
    best_x = opt[:Xi][sortperm(opt[:yi])][1:cld(n_iter, 20)]  # top 20%
    fx, i = findmin(loss.(best_x; nr=n_roll*10))

    return (theta=x2theta(best_x[i]), reward=-fx, X=opt[:Xi], y=opt[:yi])
end

# Rollouts.
function tracked_rollout(prm::Params, policy::Policy; max_steps=500, value=nothing)
    s = State(prm)
    b = Belief(s)
    if value != nothing
        s.value[:] = value
    end
    reward = 0
    focused = Int[]
    for step in 1:max_steps
        a = (step == max_steps) ? TERM : policy(b)
        reward += step!(prm, b, s, a)
        push!(focused, b.focused)
        if a == TERM
            return (value=s.value, belief=b.mu, focused=focused, choice=argmax(b.mu), reward=reward)
        end
    end
end


function do_rollouts(job::Job; exp_values=false)
    Random.seed!(0)
    prm = Params(job)
    policy = Policy(prm, Array{Float64}(load(job, :optim)["theta"]))

    if exp_values
        V = h5read("data.h5", "values")
        @time rollouts = [
            tracked_rollout(prm, policy; value=V[:, j])
            for i in 1:10
            for j in 1:size(V)[2]
        ]
    else
        @time rollouts = [
            tracked_rollout(prm, policy)
            for i in 1:job.n_sim
        ]

    end
    save(job, :matched_rollouts, rollouts)
    flush(stdout)
end


function do_optimization(job::Job)
    println("Optimization")
    flush(stdout)
    @time result = optimize(job)
    println(result.reward)
    save(job, :optim, result)
    flush(stdout)
end

function main(job::Job)
    try
        println(job)
        mkpath("runs/$(job.group)/results")
        do_optimization(job)
        # do_rollouts(job)
    finally
        for i in workers()
            i > 1 && rmprocs(i)
        end
    end
end

main(args...) = main(Job(args...))
main(file::AbstractString) = main(Job(file))

if !isempty(ARGS)
    job_group, job_id = ARGS
    main("runs/$job_group/jobs/$job_id.json")
end
