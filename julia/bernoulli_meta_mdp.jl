"""
Code for the Bernoulli meta-MDP. This is just like the main meta-MDP except
the Normal distributions are replaced by Bernoulli.

This module defines the meta-MDP, the exact solution by dynamic programming,
the VOI features, and the BMPS policy.
"""

using Distributions
using Memoize
import Random
using Parameters
import Base
using Printf
using SplitApplyCombine
# using StaticArrays
using Memoize
using QuadGK


@isdefined(⊥) || const ⊥ = 0

@with_kw struct MetaMDP
    n_arm::Int = 3
    sample_cost::Float64 = 0.001
    switch_cost::Float64 = 1
    max_obs::Int = 20
end

struct Belief
    counts::Vector{Tuple{Int, Int}}  # number of heads and tails for each item
    focused::Int
end

Belief(m::MetaMDP) = begin
    counts = [(1,1) for _ in 1:m.n_arm]
    Belief(counts, 0)
end

Base.:(==)(b1::Belief, b2::Belief) = b1.focused == b2.focused && b1.counts == b2.counts
Base.hash(b::Belief) = hash(b.focused, hash(b.counts))
Base.getindex(b::Belief, idx) = b.counts[idx]
Base.length(b::Belief) = length(b.counts)
Base.iterate(b::Belief) = iterate(b.counts)
Base.iterate(b::Belief, i) = iterate(b.counts, i)
n_obs(b::Belief) = sum(map(sum, b)) - 2length(b)
actions(m::MetaMDP, b::Belief) = n_obs(b) >= m.max_obs ? (0:0) : (0:length(b))

function Base.show(io::IO, b::Belief)
    print(io, "[ ")
    counts = map(1:length(b.counts)) do i
        h, t = b.counts[i]
        i == b.focused ? @sprintf("<%02d %02d>", h, t) : @sprintf(" %02d %02d ", h, t)
    end
    print(io, join(counts, " "))
    print(io, " ]")
end


function update(b::Belief, arm::Int, heads::Bool)::Belief
    counts = copy(b.counts)
    h, t = counts[arm]
    counts[arm] = heads ? (h+1, t) : (h, t+1)
    Belief(counts, arm)
end

function update!(b, c)
    h, t = b.counts[c]
    heads = rand() < p_heads((h, t))
    b.counts[c] = heads ? (h+1, t) : (h, t+1)
    b
end


function cost(m::MetaMDP, b::Belief, c::Int)::Float64
    m.sample_cost * (c != b.focused ? m.switch_cost : 1)
end

p_heads(counts) = counts[1] / (counts[1] + counts[2])

term_reward(b::Belief)::Float64 = maximum(p_heads.(b.counts))
is_terminal(b::Belief) = b.focused == -1
terminate(b::Belief) = Belief(b.counts, -1)

Result = Tuple{Float64, Belief, Float64}
function results(m::MetaMDP, b::Belief, c::Int)::Vector{Result}
    is_terminal(b) && error("Belief is terminal.")
    if c == ⊥
        return [(1., terminate(b), term_reward(b))]
    end
    p1 = p_heads(b.counts[c])
    p0 = 1 - p1
    r = -cost(m, b, c)
    [(p0, update(b, c, false), r),
     (p1, update(b, c, true), r)]
end

# %% ==================== Features ====================

function voi1(m::MetaMDP, b::Belief, c)
    sum(p * (term_reward(s1) + r ) for (p,s1,r) in results(m, b, c)) - term_reward(b)
end

function voi_action(m::MetaMDP, b::Belief, c)
    vals = [p_heads(counts) for counts in b.counts]
    best, second = sortperm(vals; rev=true)
    competing_val = c == best ? vals[second] : vals[best]
    h, t = b.counts[c]
    expected_max_constant(Beta(h, t), competing_val) - term_reward(b)
end

function vpi(m::MetaMDP, b::Belief)
    dists = Tuple(Beta(a...) for a in sort(b.counts))
    expected_max(dists) - term_reward(b)
end

@memoize function expected_max(dists::Tuple{Vararg{Beta}})
    mcdf(x) = mapreduce(*, dists) do d
        cdf(d, x)
    end
    quadgk(x->1-mcdf(x), 0, 1, atol=1e-8)[1]
end

@memoize function expected_max_constant(d, k)
    choose_new = quadgk(k, 1, atol=1e-8) do x
        pdf(d, x) * x
    end |> first
    choose_k = cdf(d, k) * k
    choose_new + choose_k
end


function features(m::MetaMDP, b::Belief)
    vpi_ = vpi(m, b)
    phi(c) = [
        -1,
        voi1(m, b, c),
        voi_action(m, b, c),
        vpi_,
    ]
    combinedims([phi(c) for c in 1:m.n_arm])
end

# # %% ==================== Solution ====================
function symmetry_breaking_hash(s::Belief)
    key = UInt64(0)
    for i in 1:length(s.counts)
        key += (hash(s.counts[i]) << 3(i == s.focused))
    end
    key
end

struct ValueFunction
    m::MetaMDP
    cache::Dict{UInt64, Float64}
end
ValueFunction(m::MetaMDP) = ValueFunction(m, Dict{UInt64, Float64}())

function Q(V::ValueFunction, b::Belief, c::Int)::Float64
    c == ⊥ && return term_reward(b)
    sum(p * (r + V(s1)) for (p, s1, r) in results(V.m, b, c))
end

function (V::ValueFunction)(b::Belief)::Float64
    key = symmetry_breaking_hash(b)
    haskey(V.cache, key) && return V.cache[key]
    return V.cache[key] = maximum(Q(V, b, c) for c in actions(V.m, b))
end

# %% ==================== Policy ====================
function argmaxes(f, x::AbstractArray{T})::Set{T} where T
    r = Set{T}()
    fx = f.(x)
    mfx = maximum(fx)
    for i in eachindex(x)
        if fx[i] == mfx
            push!(r, x[i])
        end
    end
    r
end
# argmax(a->Q(b,a), actions(m, b))
# policy(b::Belief) = [argmax([Q(b, a) for a in ACTIONS])]


abstract type Policy end
function act(pol::Policy, b::Belief)
    rand(actions(pol, b))
end

struct OptimalPolicy <: Policy
    m::MetaMDP
    V::ValueFunction
end
OptimalPolicy(m::MetaMDP) = OptimalPolicy(m, ValueFunction(m))
OptimalPolicy(V::ValueFunction) = OptimalPolicy(V.m, V)
(pol::OptimalPolicy)(b::Belief) = act(pol, b)

function actions(pol::OptimalPolicy, b::Belief)
    argmaxes(c->Q(pol.V, b, c), actions(pol.m, b))
end

struct BMPSPolicy <: Policy
    m::MetaMDP
    θ::Vector{Float64}
end
(pol::BMPSPolicy)(b::Belief) = act(pol, b)
BMPSPolicy(m, θ, α) = BMPSPolicy(m, θ)  # WARNING: ignoring temperature

function voc(pol, b::Belief)
    m = pol.m
    x = (pol.θ' * features(m, b))'
    x .-= [cost(m, b, c) for c in 1:m.n_arm]
    x
end

function actions(pol::BMPSPolicy, b::Belief)
    voc_ = voc(pol, b)
    argmaxes(actions(pol.m, b)) do c
        c == 0 && return 0
        voc_[c]
    end
end


function rollout(policy; b=nothing, max_steps=1000, callback=(b, c)->nothing)
    m = policy.m
    if b == nothing
        b = Belief(m)
    end
    reward = 0
    # print('x')
    max_steps = min(max_steps, m.max_obs + 1)
    for step in 1:m.max_obs+1
        c = (step == max_steps) ? ⊥ : policy(b)
        callback(b, c)
        if c == ⊥
            reward += term_reward(b)
            return (reward=reward, steps=step, belief=b)
        else
            reward -= cost(m, b, c)
            update!(b, c)
        end

    end
end

rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)

# ---------- Optimization helpers ---------- #

"Identifies the cost parameter that makes a hard-maximizing policy never take any computations."
function max_cost(m::MetaMDP)
    θ = [1., 0, 0, 1]
    b = Belief(m)
    # s = State(m)
    # b = Belief(s)
    function computes()
        pol = BMPSPolicy(m, θ)
        all(pol(b) != ⊥ for i in 1:30)
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

"Transforms a value from the 3D unit hybercube to weights for BMPS"
function x2theta(mc, x)
    # This is a trick to go from two Uniform(0,1) samples to 
    # a unifomrm sample in the 3D simplex.
    voi_weights = diff([0; sort(collect(x[2:3])); 1])
    [x[1] * mc; voi_weights]
end

