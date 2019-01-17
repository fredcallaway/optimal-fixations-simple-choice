using Distributed
@everywhere include("model.jl")
include("job.jl")
include("skopt.jl")

import Random
using Dates: now
import JSON
using LatinHypercubeSampling
using Printf



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
    computes() = SlowPolicy(m, theta)(b) != TERM

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
    m = MetaMDP(job)

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
    bounds = [ (0., max_cost(m)), (0., 1.), (0., 1.) ]
    n_latin = max(2, cld(n_iter, 4))
    opt = skopt.Optimizer(bounds, random_state=seed, n_initial_points=n_latin)
    
    # Choose initial samples (1/4) by Latin Hypersquare sampling.
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

    x1, y1 = expected_minimum(opt)

    return (X=opt[:Xi], y=opt[:yi], x1=x1, y1=y1)
end


function sum_reward(policy; n_roll=1000, seed=0)
    @distributed (+) for i in 1:n_roll
        Random.seed!(seed + i)
        rollout(policy.m, policy, max_steps=200).reward
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
