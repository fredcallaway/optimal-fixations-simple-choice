using Distributed
addprocs()
@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
end

include("gp_min.jl")
using Printf


function max_cost(m::MetaMDP)
    θ = [1., 0, 0, 1]
    s = State(m)
    b = Belief(s)
    computes() = BMPSPolicy(m, θ)(b) != TERM

    while computes()
        θ[1] *= 2
    end

    while !computes()
        θ[1] /= 2
        if θ[1] < 2^-10
            error("Computation is too expensive")
        end
    end

    step_size = θ[1] / 100
    while computes()
        θ[1] += step_size
    end
    θ[1]
end


function optimize(m::MetaMDP; n_iter=400, seed=1, n_roll=1000, verbose=false)
    Random.seed!(seed)
    mc = max_cost(m)

    function x2theta(x)
        voi_weights = collect(x)[2:end]
        voi_weights /= sum(voi_weights)
        [x[1] * mc; voi_weights]
    end

    function loss(x; nr=n_roll)
        policy = BMPSPolicy(m, x2theta(x))
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

    opt = gp_minimize(loss, 4, noisebounds=[-4, -2], iterations=n_iter; verbose=false)
    @show loss(opt.observed_optimizer, nr=10000)
    return opt
end

function optimize3(m::MetaMDP; n_iter=400, seed=1, n_roll=1000, verbose=false)
    Random.seed!(seed)
    mc = max_cost(m)

    function x2theta(x)
        voi_weights = diff([0; sort(collect(x[2:3])); 1])
        [x[1] * mc; voi_weights]
    end

    function loss(x; nr=n_roll)
        policy = BMPSPolicy(m, x2theta(x))
        reward, secs = @timed @distributed (+) for i in 1:nr
            rollout(policy, max_steps=200).reward
        end
        reward /= nr
        if verbose
            print("θ = ", round.(x2theta(x); digits=2), "   ")
            @printf "reward = %.3f   seconds = %.3f\n" reward secs
            flush(stdout)
        end
        - 10 * reward
    end

    opt = gp_minimize(loss, 3, noisebounds=[-4, -2], iterations=n_iter; verbose=false)
    @show loss(opt.observed_optimizer, nr=10000)
    return opt
end

