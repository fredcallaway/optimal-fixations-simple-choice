using OnlineStats

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

function initial_population(m, N; α=Inf, sobol=false)
    mc = max_cost(m)
    if sobol
        seq = SobolSeq(3)
        return [BMPSPolicy(m, x2theta(mc, next!(seq)), α) for i in 1:N]
    else
        g = ceil(Int, N ^ (1/3))
        @assert g ≈ N ^ (1/3)
        X = rand_grid(g)
        return [BMPSPolicy(m, x2theta(mc, X[i, :]), float(α)) for i in 1:N]
    end
end

function sample_rewards(policy, n_roll)
    map(1:n_roll) do i
        rollout(policy, max_steps=200).reward
    end
end

function ucb(m::MetaMDP; α=Inf, β::Float64=3., N::Int=8000, n_top::Int=1,
             n_roll::Int=1000, n_init::Int=100, n_iter::Int=1000)
    policies = initial_population(m, N; α=α)
    scores = [Variance() for _ in 1:N]  # tracks mean and variance
    sem = zeros(N)
    upper = zeros(N)
    μ = zeros(N)

    function pull(i; init=false)
        rewards = sample_rewards(policies[i], init ? n_init : n_roll)
        s = scores[i]
        fit!(s, rewards)
        sem[i] = √(s.σ2 / s.n)
        upper[i] = s.μ + β * sem[i]
        μ[i] = s.μ
    end

    # pull every arm once
    map(1:N) do i
        pull(i; init=true)
    end

    @debug "Begin UCB"
    hist = (pulls=Int[], top=Int[])
    best = argmax(μ)
    for t in 1:n_iter
        yield()  # for Toucher
        top = partialsortperm(upper, 1:n_top; rev=true)
        map(top) do i
            pull(i)
            push!(hist.pulls, i)
            push!(hist.top, best)
        end

        b = argmax(μ)
        if best != b
            @debug "($t) New best: $b  $(policies[b].θ)"
            best = b
        elseif t % 100 == 0
            @debug "($t)"
        end
    end

    (policies=policies, μ=μ, sem=sem, hist=hist)
end

function optimize_bmps_ucb(m::MetaMDP; kws...)
    policies, value = ucb(m; kws...)
    v, i = findmax(value)
    policies[i], v
end

