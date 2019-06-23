using Memoize
using Random
using Distributions
using SplitApplyCombine
using StaticArrays

# %% ==================== Utilities ====================

@memoize function mem_randn(d1, d2; seed=1)
    randn(MersenneTwister(seed), d1, d2)
end

@memoize mem_zeros(d1) = zeros(d1)
@memoize mem_zeros(d1, d2) = zeros(d1, d2)

"Highest value in µ not including µ[a]"
function competing_value(µ::Vector{Float64}, a::Int)
    tmp = µ[a]
    µ[a] = -Inf
    val = maximum(µ)
    µ[a] = tmp
    val
end

function expect_max_dist(d::Distribution, constant::Float64)
    p_improve = 1 - cdf(d, constant)
    p_improve < 1e-10 && return constant
    (1 - p_improve) * constant + p_improve * mean(Truncated(d, constant, Inf))
end

# %% ==================== Features ====================
function voi1_sigma(λ, obs_sigma)
    obs_λ = obs_sigma ^ -2
    w = obs_λ / (λ + obs_λ)
    sample_sigma = √(1/λ + 1/obs_λ)
    w * sample_sigma
end

function voi1(b::Belief, c::Computation)
    cv = competing_value(b.µ, c)
    d = Normal(b.µ[c], voi1_sigma(b.λ[c], b.obs_sigma))
    expect_max_dist(d, cv) - maximum(b.µ)
end

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
    σ::Float64  # running empirical std
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


# %% ==================== The Policy ====================

Weights = NamedTuple{(:cost, :voi1, :voi_action, :vpi),Tuple{Float64,Float64,Float64,Float64}}
"A metalevel policy that uses the BMPS features"
struct BMPSPolicy <: Policy
    m::MetaMDP
    θ::Weights
end
BMPSPolicy(m::MetaMDP, θ) = BMPSPolicy(m, Weights(θ))

"Selects a computation to perform in a given belief."
(pol::BMPSPolicy)(b::Belief) = fast_act(pol, b)

"VOC without VPI feature"
function fast_voc(pol::BMPSPolicy, b::Belief)
    θ = pol.θ
    map(1:pol.m.n_arm) do c
        -cost(pol.m, b, c) +
        -θ.cost +
        θ.voi1 * voi1(b, c) +
        θ.voi_action * voi_action(b, c)
    end
end

function full_voc(pol::BMPSPolicy, b::Belief, vpi_samples=10000)
    fast_voc(pol, b) .+ pol.θ.vpi * vpi(b, vpi_samples)[1]
end

function fast_act(pol::BMPSPolicy, b::Belief)
    θ = pol.θ
    voc = fast_voc(pol, b)
    v, c = findmax(noisy(voc))
    v > 0 && return c

    # No computation is good enough without VPI.
    # Try putting VPI weight on VOI_action (a lower bound on VPI)
    v + θ.vpi * voi_action(b, c) > 0 && return c

    # Still no luck. Try actual VPI. To make the VPI estimation as fast and accurate as possible,
    # we don't use a fixed number of samples. Instead, we continually add samples until
    # the estimate is precise "enough". The estimate is precise enough when:
    # (1) the uncertainty point (VOC=0) is not within 3 standard errors, OR
    # (2) the standard error is less than 1e-4
    # (3) 100000 samples of the VPI have been taken

    θ.vpi == 0. && return TERM  # no weight on VPI, VOC can't improve
    vpi = VPI(b)
    for i in 1:100000
        step!(vpi, 500)  # add 500 samples
        μ_voc = v + θ.vpi * vpi.μ
        σ_voc = θ.vpi * (vpi.σ / √vpi.n)
        (σ_voc < 1e-4 || abs(μ_voc - 0) > 3 * σ_voc) && break
        # (i == 100000) && println("Warning: VPI estimation did not converge.")
    end
    v + θ.vpi * vpi.μ > 0 && return c
    return TERM
end