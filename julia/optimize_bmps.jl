using Distributed
using SharedArrays
using Printf
using Sobol
using OnlineStats
using Statistics

function max_cost(m::MetaMDP)
    θ = [1., 0, 0, 1]
    s = State(m)
    b = Belief(s)
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

function initial_population(m, N; sobol=true)
    mc = max_cost(m)
    seq = SobolSeq(N)
    policies = [BMPSPolicy(m, x2theta(mc, next!(seq))) for i in 1:N]
    # policies = [BMPSPolicy(m, x2theta(rand(3))) for i in 1:N]
    sort!(policies, by=x->-x.θ.vpi)  # for parallel efficiency
end

# %% ====================  ====================

function ucb(m::MetaMDP; β::Float64=2., N::Int=100, n_roll::Int=100, n_init::Int = 1, n_iter::Int=1000)
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
        pessimistic_best_value = μ[best] - β * sem[best]
        if sum(pessimistic_best_value .<= upper) == 1  # only upper[best] is larger
            converged = true
            @info "Converged" t μ[best] policies[best].θ
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
    policies, fitness, n = halving(m; kws...)
    fit, i = findmax(fitness)
    policies[i], fit
end


















