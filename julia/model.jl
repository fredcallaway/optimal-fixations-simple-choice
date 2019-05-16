using Distributions
using Memoize
using Random
using Parameters
import Base
using Cuba
using SplitApplyCombine
using StaticArrays
# ---------- MetaMDP Model ---------- #

const TERM = 0
const Computation = Int

@with_kw struct MetaMDP
    n_arm::Int = 3
    obs_sigma::Float64 = 1
    sample_cost::Float64 = 0.001
    switch_cost::Float64 = 1
end

struct State
    value::Vector{Float64}
    obs_sigma::Float64
end
State(m::MetaMDP, value) = State(value, m.obs_sigma)
State(m::MetaMDP) = State(m, randn(m.n_arm))

mutable struct Belief
    mu::Vector{Float64}
    lam::Vector{Float64}
    obs_sigma::Float64
    focused::Int
end

Belief(s::State) = Belief(
    zeros(length(s.value)),
    ones(length(s.value)),
    s.obs_sigma,
    rand(1:length(s.value))
)
Base.copy(b::Belief) = Belief(copy(b.mu), copy(b.lam), b.obs_sigma, b.focused)

function step!(m::MetaMDP, b::Belief, s::State, c::Computation)
    if c == TERM
        return maximum(b.mu)
    end
    r = -cost(m, b, c)
    b.focused = c
    obs = s.value[c] + randn() * s.obs_sigma
    update!(b, c, obs)
    return r
end

function cost(m::MetaMDP, b::Belief, c::Computation)
    m.sample_cost * (b.focused == c ? 1. : m.switch_cost)
end

function update!(b::Belief, c::Computation, obs)
    obs_lam = b.obs_sigma ^ -2
    lam1 = b.lam[c] + obs_lam
    mu1 = (obs * obs_lam + b.mu[c] * b.lam[c]) / lam1
    b.mu[c] = mu1
    b.lam[c] = lam1
end


# ---------- BMPS Features ---------- #

@memoize function mem_randn(d1, d2; seed=1)
    randn(MersenneTwister(seed), d1, d2)
end

@memoize mem_zeros(d1) = zeros(d1)
@memoize mem_zeros(d1, d2) = zeros(d1, d2)

"Highest value in mu not including mu[a]"
function competing_value(mu::Vector{Float64}, a::Int)
    tmp = mu[a]
    mu[a] = -Inf
    val = maximum(mu)
    mu[a] = tmp
    val
end

function expect_max_dist(d::Distribution, constant::Float64)
    p_improve = 1 - cdf(d, constant)
    p_improve < 1e-10 && return constant
    (1 - p_improve) * constant + p_improve * mean(Truncated(d, constant, Inf))
end

function voi_action(b::Belief, a::Int)
    cv = competing_value(b.mu, a)
    d = Normal(b.mu[a], b.lam[a] ^ -0.5)
    expect_max_dist(d, cv) - maximum(b.mu)
end

function voi1_sigma(lam, obs_sigma)
    obs_lam = obs_sigma ^ -2
    w = obs_lam / (lam + obs_lam)
    sample_sigma = √(1/lam + 1/obs_lam)
    w * sample_sigma
end

function voi1(b::Belief, c::Computation)
    cv = competing_value(b.mu, c)
    d = Normal(b.mu[c], voi1_sigma(b.lam[c], b.obs_sigma))
    expect_max_dist(d, cv) - maximum(b.mu)
end

function vpi2(b::Belief, n_sample)
    R = randn!(mem_zeros(n_sample, length(b.mu)))
    R .*= (b.lam .^ -0.5)' .+ b.mu'
    max_samples = maximum!(mem_zeros(n_sample), R)
    mean(max_samples) - maximum(b.mu), std(max_samples)
end

mutable struct VPI
    b::Belief
    µ::Float64
    σ::Float64
    n::Int
end
VPI(b::Belief) = VPI(b, 0., 0., 0.)

function step!(v::VPI, n_sample=100)
    n1 = v.n + n_sample
    μ, σ = vpi2(v.b, n_sample)
    v.μ = v.n/n1 * v.μ + n_sample/n1 * μ
    v.σ = v.n/n1 * v.σ + n_sample/n1 * σ
    v.n = n1
end
# function vpi(b)
#     μ, σ = b.mu, b.lam .^ -0.5
#     dists = Normal.(μ, σ)
#     low, high = μ .- (5 .* σ), μ .+ (5 .* σ)
#     mult = prod(high - low)
#     g(x, v) = begin
#         x .= low .+ (high-low) .* x
#         v .= maximum(x; dims=1) .* prod(pdf.(dists, x); dims=1) .* mult
#     end
#      # hcubature(g, zeros(3), ones(3), atol=1e-4);
#      cuhre(g, 3, nvec=1000).integral[1] - maximum(μ)
# end
# function vpi(b; n_sample=50000)
#     # Use pre-allocated arrays efficiency
#     R = mem_zeros(n_sample, length(b.mu))
#     max_samples = mem_zeros(n_sample)
#
#     copyto!(R, mem_randn(n_sample, length(b.mu)))
#     R .*= (b.lam .^ -0.5)' .+ b.mu'
#
#     maximum!(max_samples, R)
#     mean(max_samples) - maximum(b.mu)
# end

n_obs(m::MetaMDP, b::Belief) = Int(round((sum(b.lam) - m.n_arm) * m.obs_sigma ^ 2))

function features(m::MetaMDP, b::Belief)
    vpi_ = vpi(b)

    phi(c) = [
        -1,
        -(b.focused != c),
        voi1(b, c),
        voi_action(b, c),
        vpi_,
    ]
    combinedims([phi(c) for c in 1:m.n_arm])
end


# ---------- Policy ---------- #
noisy(x, ε=1e-10) = x .+ ε .* rand(length(x))

"A metalevel policy that uses the BMPS features"
struct Policy
    m::MetaMDP
    θ::SVector{6,Float64}
end
"Selects a computation to perform in a given belief."
(π::Policy)(b::Belief; slow=false) = (slow ? slow_act : fast_act)(π, b)

function slow_act(π::Policy, b::Belief)
    voc = (π.θ' * features(π.m, b))'
    voc .-= [cost(π.m, b, c) for c in 1:π.m.n_arm]
    v, c = findmax(noisy(voc))
    v <= 0 ? TERM : c
end

function fast_voc(π::Policy, b::Belief)
    θ = π.θ
    map(1:π.m.n_arm) do c
        -cost(π.m, b, c) +  # current cost
        -θ[1] +  # future cost
        -θ[2] * max(0, (1 - θ[3] * n_obs(π.m, b))) * (b.focused != c) +  # extra switch penalty
        θ[4] * voi1(b, c) + θ[5] * voi_action(b, c)  # value of information
    end
end

function voc(π::Policy, b::Belief)
    fast_voc(π, b) .+ π.θ[6] * vpi(b)
end

function fast_act(π::Policy, b::Belief)
    voc = fast_voc(π, b)
    v, c = findmax(noisy(voc))
    v > 0 && return c
    # No computation is good enough without VPI.
    # Try putting VPI weight on VOI_action (a lower bound on VPI)
    v + π.θ[6] * voi_action(b, c) > 0 && return c

    # Still no luck. Try VPI.
    π.θ[6] == 0. && return TERM  # skip VPI
    vpi = VPI(b)
    for i in 1:100000  # Iteratively refine estimate until confident in choice.
        step!(vpi, 500)
        μ_v = v + π.θ[6] * vpi.μ
        sem_v = π.θ[6] * (vpi.σ / √vpi.n)
        (sem_v < 0.0001 || abs(μ_v - 0) > 3sem_v) && break
        # (i == 100000) && println("Warning: VPI estimation did not converge.")
    end
    v + π.θ[6] * vpi.μ > 0 && return c
    return TERM
end


struct MetaGreedy
    m::MetaMDP
end

(π::MetaGreedy)(b::Belief) = begin
    voc1 = [voi1(b, c) - cost(π.m, b, c) for c in 1:π.m.n_arm]
    v, c = findmax(noisy(voc1))
    v <= 0 ? TERM : c
end

struct Noisy{T}
    ε::Float64
    π::T
    m::MetaMDP
end
Noisy(ε, π) = Noisy(ε, π, π.m)

(π::Noisy)(b::Belief) = begin
    rand() < π.ε ? rand(1:length(b.mu)) : π.π(b)
end


function rollout(policy; state=nothing, max_steps=1000, callback=(b, c)->nothing)
    m = policy.m
    s = state == nothing ? State(m) : state
    b = Belief(s)
    reward = 0
    # print('x')
    for step in 1:max_steps
        c = (step == max_steps) ? TERM : policy(b)
        callback(b, c)
        reward += step!(m, b, s, c)
        if c == TERM
            return (reward=reward, choice=argmax(noisy(b.mu)), steps=step, belief=b)
        end
    end
end

rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)
