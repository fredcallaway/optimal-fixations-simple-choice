using Distributed
@everywhere include("model.jl")
include("job.jl")
include("skopt.jl")

import Random
using Dates: now
import JSON
using LatinHypercubeSampling
using Printf

function max_cost(m::MetaMDP)
    theta = [1., 0, 0, 0, 0, 1]
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
    @unpack seed, n_iter, n_roll, cost_features = job
    m = MetaMDP(job)
    optimize(m, n_iter=n_iter, n_roll=n_roll, seed=seed,
             cost_features=cost_features, verbose=verbose)
end

function optimize(m::MetaMDP; cost_features=1, n_iter=200, seed=1, n_roll=1000, verbose=true)
    Random.seed!(seed)
    function x2theta(x)
        x = collect(x)
        cost_weights = x[1:cost_features]
        voi_weights = diff([0; sort(collect(x[end-1:end])); 1])
        θ = [cost_weights; zeros(3-cost_features); voi_weights]
        if cost_features == 3
            θ[3] = exp(θ[3])
        end
        θ
    end

    function loss(x; nr=n_roll)
        policy = Policy(m, x2theta(x))
        reward, secs = @timed @distributed (+) for i in 1:nr
            rollout(policy, max_steps=200).reward
        end
        reward /= nr
        if verbose
            print("θ = ", round.(x2theta(x); digits=2), "   ")
            @printf "reward = %.3f   seconds = %.3f\n" reward secs
            flush(stdout)
        end
        - reward
    end

    mc = max_cost(m)
    bounds = [
        [(0., mc), (0., mc), (-8., -1.)][1:cost_features] ;
        [(0., 1.), (0., 1.)]
    ]
    n_latin = max(2, cld(n_iter, 4))
    opt = skopt.Optimizer(bounds, random_state=seed, n_initial_points=n_latin)

    # Choose initial samples (1/4) by Latin Hypersquare sampling.
    lb, ub = invert(bounds)
    db = ub .- lb
    rescale(x) = lb .+ (x ./ n_latin) .* db
    latin_points = LHCoptim(n_latin, length(bounds), 1000)[1]
    @sync for i in 1:n_latin
        x = rescale(latin_points[i, :])
        @async tell(opt, x, loss(x))
    end

    # Bayesian optimization.
    println("Begin Bayesian optimization.")
    for i in 1:(n_iter - n_latin)
        x = ask(opt)
        tell(opt, x, loss(x))
    end

    x1, y1 = expected_minimum(opt)
    θ1 = x2theta(x1)
    println("Exepected best: ", round.(θ1; digits=2), "  ", -round(y1; digits=3))
    return (θ_i=x2theta.(opt.Xi), r_i=-opt.yi, θ1=θ1, r1=-y1)
end


function sum_reward(policy; n_roll=1000, seed=0)
    @distributed (+) for i in 1:n_roll
        Random.seed!(seed + i)
        rollout(policy, max_steps=200).reward
    end
end


function halving(m::MetaMDP)
    n = 500
    bounds = [ (0., max_cost(m)), (0., 1.), (0., 1.) ]
    pop = [Policy(m, x2theta(rand(3) .* [b[2] for b in bounds])) for i in 1:n]

    n_eval = zeros(Int, n)
    score = zeros(n)
    avg_score = zeros(n)

    reduction = 2
    for t in 0:7
        r = 100 * reduction ^ t
        q = 1 - 1 / reduction ^ t
        active = avg_score .>= quantile(avg_score, q)
        for i in 1:n
            if active[i]
                score[i] += sum_reward(pop[i]; n_roll=r, seed=n_eval[i])
                n_eval[i] += r
            end
        end
        avg_score = score ./ n_eval
        println(maximum(avg_score), "  ", sum(active))
    end
    return (pop=[p.θ for p in pop], avg_score=avg_score, n_eval=n_eval)
end


function main(job::Job)
    try
        println("Start ", now())
        mkpath("runs/$(job.group)/results")
        println("Optimization")
        flush(stdout)
        @time result = optimize(job)
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
