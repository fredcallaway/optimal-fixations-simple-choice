using Distributed
include("bernoulli_metabandits.jl")
# include("job.jl")
include("skopt.jl")

import Random
using Dates: now
import JSON
using LatinHypercubeSampling
using Printf

function max_cost(m::MetaMDP)
    theta = [1., 0, 0, 1]
    b = Belief(m)
    computes() = BMPSPolicy(m, theta)(b) != TERM_ACTION

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

function x2theta(x)
    x = collect(x)
    voi_weights = diff([0; sort(x[2:3]); 1])
    [x[1]; voi_weights]
end

function optimize(m::MetaMDP; seed=1, n_iter=200, n_roll=1000, verbose=true)
    Random.seed!(seed)


    function loss(x; nr=n_roll)
        policy = BMPSPolicy(m, x2theta(x))
        reward, secs = @timed mapreduce(+, 1:nr) do i
        # reward, secs = @timed @distributed (+) for i in 1:nr
            rollout(policy, max_steps=50).reward
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
    bounds = [(0., mc), (0., 1.), (0., 1.)]
    n_latin = max(2, cld(n_iter, 4))
    opt = skopt.Optimizer(bounds, random_state=seed, n_initial_points=n_latin)

    # Choose initial samples (1/4) by Latin Hypersquare sampling.
    lb, ub = invert(bounds)
    db = ub .- lb
    rescale(x) = lb .+ (x ./ n_latin) .* db
    latin_points = LHCoptim(n_latin, length(bounds), 1000)[1]
    for i in 1:n_latin
        x = rescale(latin_points[i, :])
        tell(opt, x, loss(x))
    end

    # Bayesian optimization.
    verbose && println("Begin Bayesian optimization.")
    for i in 1:(n_iter - n_latin)
        x = ask(opt)
        tell(opt, x, loss(x))
    end

    x1, y1 = expected_minimum(opt)
    θ1 = x2theta(collect(x1))
    println("Expected best: ", round.(θ1; digits=2), "  ", -round(y1; digits=3))
    return (θ_i=x2theta.(opt.Xi), r_i=-opt.yi, θ1=θ1, r1=-y1)
end

# function main(job::Job)
#     try
#         println("Start ", now())
#         mkpath("runs/$(job.group)/results")
#         println("Optimization")
#         flush(stdout)
#         @time result = optimize(job)
#         save(job, :optim, result)
#         println("Done ", now())
#         flush(stdout)
#     finally
#         for i in workers()
#             i > 1 && rmprocs(i)
#         end
#     end
# end

# main(args...) = main(Job(args...))
# main(file::AbstractString) = main(Job(file))



# if !isempty(ARGS)
#     job_group, job_id = ARGS
#     main("runs/$job_group/jobs/$job_id.json")
# end
