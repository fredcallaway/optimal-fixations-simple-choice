using Memoize
using Distributions
using Random
using QuadGK

using StatsFuns: normcdf, normpdf
Φ = normcdf
ϕ = normpdf
# %% ==================== Utilities ====================

@memoize mem_zeros(dims...) = zeros(dims...)

"Highest value in µ not including µ[a]"
function competing_value(µ::Vector{Float64}, a::Int)
    tmp = µ[a]
    µ[a] = -Inf
    val = maximum(µ)
    µ[a] = tmp
    val
end

"Expected maximum of a distribution and a constant"
function expect_max_dist(d::Distribution, constant::Float64)
    p_improve = 1 - cdf(d, constant)
    p_improve < 1e-10 && return constant
    (1 - p_improve) * constant + p_improve * mean(Truncated(d, constant, Inf))
end

"Standard deviation of the posterior mean"
function std_of_posterior_mean(λ, σ_obs)
    obs_λ = σ_obs ^ -2
    w = obs_λ / (λ + obs_λ)
    sample_sigma = √(1/λ + 1/obs_λ)
    w * sample_sigma
end

# %% ==================== Features ====================

"Value of information from n samples of item c"
function voi_n(b::Belief, c::Computation, n::Int)
    cv = competing_value(b.µ, c)
    σ_μ = std_of_posterior_mean(b.λ[c], b.σ_obs / √n)
    σ_μ ≈ 0. && return 0.  # avoid error initializing Normal
    d = Normal(b.µ[c], σ_μ)
    expect_max_dist(d, cv) - maximum(b.µ)
end

"Myopic value of information"
voi1(b, c) = voi_n(b, c, 1)

"Value of perfect information about one action"
function voi_action(b::Belief, a::Int)
    cv = competing_value(b.µ, a)
    d = Normal(b.µ[a], b.λ[a] ^ -0.5)
    expect_max_dist(d, cv) - maximum(b.µ)
end

"Expected maximum of Normals with means μ and precisions λ"
function expected_max_norm(μ, λ)
    if length(μ) == 2
        μ1, μ2 = μ
        σ1, σ2 = λ .^ -0.5
        θ = √(σ1^2 + σ2^2)
        return μ1 * Φ((μ1 - μ2) / θ) + μ2 * Φ((μ2 - μ1) / θ) + θ * ϕ((μ1 - μ2) / θ)
    end

    dists = Normal.(μ, λ.^-0.5)
    mcdf(x) = mapreduce(*, dists) do d
        cdf(d, x)
    end

    - quadgk(mcdf, -10, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, 10, atol=1e-5)[1]
end

"Value of perfect information about all items"
function vpi(b)
    expected_max_norm(b.μ, b.λ) - maximum(b.μ)
end

"VPI approximated with Monte Carlo instead of numerical integration"
function vpi_montecarlo(b::Belief, n_sample)
    R = randn!(mem_zeros(n_sample, length(b.µ)))
    @. R = R * (b.λ ^ -0.5)' + b.μ'
    max_samples = maximum!(mem_zeros(n_sample), R)
    mean(max_samples) - maximum(b.µ), std(max_samples)
end
