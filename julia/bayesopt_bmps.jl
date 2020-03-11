using Distributed
using Printf
using Sobol
using Statistics
include("gp_min.jl")

function mean_reward(policy, n_roll, parallel)
    if parallel
        rr = @distributed (+) for i in 1:n_roll
            rollout(policy, max_steps=200).reward
        end
        return rr / n_roll
    else
        rr = mapreduce(+, 1:n_roll) do i
            rollout(policy, max_steps=200).reward
        end
        return rr / n_roll
    end

end


function optimize_bmps(m::MetaMDP; α=Inf, n_iter=500, seed=nothing, n_roll=10000,
                  verbose=false, parallel=true, repetitions=1)
    if seed != nothing
        Random.seed!(seed)
    end
    mc = max_cost(m)

    function loss(x, nr=n_roll)
        policy = BMPSPolicy(m, x2theta(mc, x), α)
        reward, secs = @timed mean_reward(policy, n_roll, parallel)
        if verbose
            print("θ = ", round.(x2theta(mc, x); digits=2), "   ")
            @printf "reward = %.3f   seconds = %.3f\n" reward secs
            flush(stdout)
        end
        -reward
    end

    opt = gp_minimize(loss, 3, noisebounds=[-4, -2],
                      iterations=n_iter, repetitions=repetitions,
                      verbose=false)

    f_mod = loss(opt.model_optimizer, 10 * n_roll)
    f_obs = loss(opt.observed_optimizer, 10 * n_roll)
    best = f_obs < f_mod ? opt.observed_optimizer : opt.model_optimizer
    return BMPSPolicy(m, x2theta(mc, best), α)
end
