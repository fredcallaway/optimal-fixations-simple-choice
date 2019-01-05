using Distributed
@everywhere include("model.jl")
include("job.jl")

import Random
using Dates: now
using PyCall
import JSON
using LatinHypercubeSampling
using Printf

@pyimport numpy
@pyimport skopt
# Optimizer methods
ask(opt)::Vector{Float64} = opt[:ask]()
tell(opt, x::Vector{Float64}, fx::Float64) = opt[:tell](Tuple(x), fx)

# Optimization.
function x2theta(x)
    cost_weight = x[1]
    voi_weights = diff([0; sort(collect(x[2:end])); 1])
    [cost_weight; voi_weights]
end

function max_cost(m::MetaMDP)
    theta = Float64[1, 0, 0, 1]
    s = State(m)
    b = Belief(s)
    computes() = Policy(m, theta)(b) != TERM

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

function optimize(job::Job; verbose=true)
    @unpack seed, n_iter, n_roll = job
    Random.seed!(seed)
    numpy.random[:seed](seed)
    m = MetaMDP(job)

    function loss(x; nr=n_roll)
        policy = Policy(m, x2theta(x))
        reward, secs = @timed @distributed (+) for i in 1:nr
            rollout(m, policy, max_steps=200).reward
        end
        reward /= nr
        if verbose
            @printf "reward = %.3f   seconds = %.3f\n" reward secs
            flush(stdout)
        end
        - reward
    end
    bounds = [ (0., max_cost(m)), (0., 1.), (0., 1.) ]
    n_latin = max(2, cld(n_iter, 4))
    opt = skopt.Optimizer(bounds, random_state=seed, n_initial_points=n_latin)


    # Choose initial samples by Latin Hypersquare sampling.
    upper_bounds = [b[2] for b in bounds]
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

    println("Cross validation.")
    best_x = opt[:Xi][sortperm(opt[:yi])][1:cld(n_iter, 5)]  # top 20%
    fx, i = findmin(loss.(best_x; nr=n_roll*10))

    return (theta=x2theta(best_x[i]), reward=-fx, X=opt[:Xi], y=opt[:yi])
end

function main(job::Job)
    try
        println("Start ", now())
        mkpath("runs/$(job.group)/results")
        println("Optimization")
        flush(stdout)
        @time result = optimize(job)
        println(result.reward)
        save(job, :optim, result)
        println("Done ", now())
        flush(stdout)
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
