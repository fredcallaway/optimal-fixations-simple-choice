using Parameters
@isdefined(⊥) || const ⊥ = 0  # terminal action
@isdefined(Computation) || const Computation = Int

noisy(x, ε=1e-10) = x .+ ε .* rand(length(x))

"Metalevel Markov decision process"
@with_kw struct MetaMDP
    n_arm::Int = 3                 # number of items to choose between
    σ_obs::Float64 = 1             # std of observation distribution
    sample_cost::Float64 = 0.001   # cost per sample
    switch_cost::Float64 = 0.      # additional cost for sampling a different item
end

"Base type for metalevel policies."
abstract type Policy end


"Ground truth state"
struct State
    value::Vector{Float64}  # values of each item in the choice set
    σ_obs::Float64          # MetaMDP parameter, stored here for convenience
end
State(m::MetaMDP, value) = State(value, m.σ_obs)
State(m::MetaMDP) = State(m, randn(m.n_arm))


"Belief state"
mutable struct Belief
    µ::Vector{Float64}  # mean vector
    λ::Vector{Float64}  # precision vector
    σ_obs::Float64      # metaMDP parameter, stored here for convenience
    focused::Int        # currently fixated item (necessary for switch cost)
end

Belief(s::State) = Belief(
    zeros(length(s.value)),
    ones(length(s.value)),
    s.σ_obs,
    0
)
Belief(m::MetaMDP) = Belief(
    zeros(m.n_arm),
    ones(m.n_arm),
    m.σ_obs,
    0
)
Base.copy(b::Belief) = Belief(copy(b.µ), copy(b.λ), b.σ_obs, b.focused)


"""Expected reward for making a decision.

We use the expected reward rather than the ground truth value because it has the same
expectation but lower variance. This makes learning more efficient, but dosen't
introduce any bias or change the optimal policy. See https://arxiv.org/abs/1408.2048
"""
function term_reward(b::Belief)
    maximum(b.µ)
end

"Sampling cost function, includes switching cost."
function cost(m::MetaMDP, b::Belief, c::Computation)
    if b.focused != 0 && b.focused != c
        return m.sample_cost + m.switch_cost
    else
        return m.sample_cost
    end
end

"Updates belief based on the given computation."
function transition!(b::Belief, s::State, c::Computation)
    b.focused = c
    obs = s.value[c] + randn() * s.σ_obs
    b.µ[c], b.λ[c] = bayes_update_normal(b.μ[c], b.λ[c], obs, s.σ_obs ^ -2)
end

"Returns updated mean and precision given a prior and observation."
function bayes_update_normal(μ, λ, obs, λ_obs)
    λ1 = λ + λ_obs
    μ1 = (obs * λ_obs + μ * λ) / λ1
    (μ1, λ1)
end

"Run one rollout (one decision) of a policy on its associated MetaMDP."
function rollout(policy::Policy; state=nothing, max_steps=1000, callback=(b, c)->nothing)
    m = policy.m
    s = state == nothing ? State(m) : state
    b = Belief(s)
    reward = 0
    for t in 1:max_steps
        c = (t == max_steps) ? ⊥ : policy(b)
        callback(b, c)
        if c == ⊥
            reward += term_reward(b)
            return (reward=reward, choice=argmax(noisy(b.µ)), steps=t, state=s, belief=b)
        else
            reward -= cost(m, b, c)
            transition!(b, s, c)
        end
    end
end

# for do block syntax
rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)

struct RandomPolicy <: Policy
    m::MetaMDP
end
(pol::RandomPolicy)(b::Belief) = rand(0:pol.m.n_arm)
