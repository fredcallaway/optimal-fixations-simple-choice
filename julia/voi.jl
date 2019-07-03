using Memoize
using Distributions
using Random

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

function voi_n(b::Belief, c::Computation, n::Int)
    cv = competing_value(b.µ, c)
    d = Normal(b.µ[c], std_of_posterior_mean(b.λ[c], b.σ_obs / √n))
    expect_max_dist(d, cv) - maximum(b.µ)
end

voi1(b, c) = voi_n(b, c, 1)

function voi_action(b::Belief, a::Int)
    cv = competing_value(b.µ, a)
    d = Normal(b.µ[a], b.λ[a] ^ -0.5)
    expect_max_dist(d, cv) - maximum(b.µ)
end

function vpi(b::Belief, n_sample)
    R = randn!(mem_zeros(n_sample, length(b.µ)))
    R .*= (b.λ .^ -0.5)' .+ b.µ'
    max_samples = maximum!(mem_zeros(n_sample), R)
    mean(max_samples) - maximum(b.µ), std(max_samples)
end

"A structure to store increasingly precise VPI estimates."
mutable struct VPI
    b::Belief
    µ::Float64  # running empirical mean
    σ::Float64  # running empirical std (sort of)
    n::Int      # number of samples
end
VPI(b::Belief) = VPI(b, 0., 0., 0)

function step!(v::VPI, n_sample=100)
    n1 = v.n + n_sample
    μ, σ = vpi(v.b, n_sample)
    v.μ = v.n/n1 * v.μ + n_sample/n1 * μ
    v.σ = v.n/n1 * v.σ + n_sample/n1 * σ
    v.n = n1
end

