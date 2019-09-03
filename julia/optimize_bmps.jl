using Distributed
using SharedArrays
using Printf
using Sobol
using OnlineStats
using Statistics

function max_cost(m::MetaMDP)
    θ = [1., 0, 0, 1]
    b = Belief(m)
    # s = State(m)
    # b = Belief(s)
    function computes()
        pol = BMPSPolicy(m, θ)
        all(pol(b) != TERM for i in 1:30)
    end

    while computes()
        θ[1] *= 2
    end

    while !computes()
        θ[1] /= 2
        if θ[1] < 2^-10
            error("Computation is too expensive")
        end
    end

    step_size = θ[1] / 10
    while computes()
        θ[1] += step_size
    end
    θ[1]
end

function x2theta(mc, x)
    voi_weights = diff([0; sort(collect(x[2:3])); 1])
    [x[1] * mc; voi_weights]
end


function rand_grid(g)
    dim = 3
    s = 1. / g
    x = 0:s:1-s
    X = zeros(g^dim, dim)

    for (i, a) in enumerate(Iterators.product(x, x, x))
        X[i, :] = collect(a) + rand(dim) * s
    end
    X
end

function initial_population(m, N; sobol=true)
    mc = max_cost(m)
    if sobol
        seq = SobolSeq(3)
        return [BMPSPolicy(m, x2theta(mc, next!(seq))) for i in 1:N]
    else
        g = ceil(Int, N ^ (1/3))
        @assert g ≈ N ^ (1/3)
        X = rand_grid(g)
        return [BMPSPolicy(m, x2theta(mc, X[i, :])) for i in 1:N]
    end
end

# %% ====================  ====================

function ucb(m::MetaMDP; β::Float64=2., N::Int=8000, n_roll::Int=64, n_init::Int = 1, n_iter::Int=1000)
    policies = initial_population(m, N)
    scores = [Variance() for _ in 1:N]  # tracks mean and variance
    sem = zeros(N)
    upper = zeros(N)
    μ = zeros(N)

    # v = SharedArray{Float64}(n_roll)
    v = zeros(n_roll)
    function pull(i)
        # @sync @distributed for j in eachindex(v)
        #     v[j] = rollout(policies[i], max_steps=200).reward
        # end
        for j in eachindex(v)
            v[j] = rollout(policies[i], max_steps=200).reward
        end
        s = scores[i]
        fit!(s, v)
        sem[i] = √(s.σ2 / s.n)
        upper[i] = s.μ + β * sem[i]
        μ[i] = s.μ
    end

    # pull every arm once
    for i in 1:N
        for _ in 1:n_init
            pull(i)
        end
    end

    hist = (pulls=Int[], top=Int[])
    best = argmax(μ)
    converged = false
    for t in 1:n_iter
        i = argmax(upper)
        pull(i)
        push!(hist.pulls, i)
        push!(hist.top, best)
        b = argmax(μ)
        if best != b
            @debug "($t) New best: $b  $(policies[b].θ)"
            best = b
        end
        converged = (sum(μ[best] .<= upper) == 1) &&
                    (sum((μ[best] - β * sem[best]) .< μ) == 1)

        if converged
            @info "Converged" t μ[best] θ=repr(policies[best].θ)
            break
        end
        # @printf "%d %.3f ± %.4f\n" i scores[i].μ sem[i]
    end
    if !converged
        @warn "UCB optimization did not converge."
    end

    (policies=policies, μ=μ, sem=sem, hist=hist, converged=converged)
end

function optimize_bmps(m::MetaMDP; kws...)
    policies, value = ucb(m; kws...)
    v, i = findmax(value)
    policies[i], v
end











