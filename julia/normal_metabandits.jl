using Distributions
using Memoize
import Random
using Parameters
import Base
using StaticArrays

@with_kw struct MetaMDP{N}
    σ_obs::Float64 = 1
    sample_cost::Float64 = 0.001
    switch_cost::Float64 = 1
    quantization::Int = 3
    μ_digits::Int = -1
end
n_arm(m::MetaMDP{N}) where {N} = N
actions(m) = 0:n_arm(m)
MetaMDP() = MetaMDP{3}()  # 3 arms by default

struct Belief{N}
    value::SVector{N,Normal{Float64}}
    focused::Int
end

Belief(m::MetaMDP{N}) where {N} = begin
    value = SVector{N}([Normal(0,1) for i in 1:N])
    Belief(value, 0)
end

Base.:(==)(b1::Belief, b2::Belief) = b1.focused == b2.focused && b1.value == b2.value
Base.hash(b::Belief) = hash(b.focused, hash(b.value))
Base.getindex(b::Belief, idx) = b.value[idx]
Base.length(b::Belief) = length(b.value)
Base.iterate(b::Belief) = iterate(b.value)
Base.iterate(b::Belief, i) = iterate(b.value, i)

Base.show(io::IO, b::Belief) = begin
    x = map(b.value) do d
        μ, σ = round.((d.μ, d.σ); digits=3)
        ("N($μ, $σ)")
    end
    print(io, "[ ", join(x, ", "), " ] ", b.focused)
end

@isdefined(⊥_ACTION) || const ⊥_ACTION = 0


@memoize function discretized_randn(q)
    d = Normal(0, 1)
    bounds = range(-3, stop=3, length=q+1)
    pieces = [Truncated(d, bounds[i:i+1]...) for i in 1:q]
    x = mean.(pieces)
    p = diff(cdf.(d, bounds))
    zip(p, x)
end
# @isdefined(DRAND) || const DRAND = discretized_randn()


function posterior(m::MetaMDP, prior::Normal)
    λ, λ_obs = (prior.σ, m.σ_obs) .^ -2;
    sample_weight = λ_obs / (λ + λ_obs)
    sample_σ = √(1/λ + 1/λ_obs)  # uncertainty about true mean plus sampling variance
    (μ_dist = Normal(prior.μ, sample_weight * sample_σ),
     σ = (λ + λ_obs) ^ -0.5)
end

function update(m::MetaMDP{N}, b::Belief, c::Int, sample::Float64=randn())::Belief where {N}
    v = collect(b.value)
    μ_dist, σ = posterior(m, b[c])
    μ = μ_dist.μ + μ_dist.σ * sample
    if m.μ_digts != -1
        μ = round(μ; digits=m.μ_digits)
    end
    Belief{N}(setindex(b.value, Normal(µ, σ), c), c)
end

function cost(m::MetaMDP, b::Belief, c::Int)::Float64
    m.sample_cost * (c != b.focused ? m.switch_cost : 1)
end

term_reward(b::Belief)::Float64 = maximum(d.μ for d in b)
is_terminal(b::Belief) = b.focused == -1
terminate(b::Belief) = Belief(b.value, -1)

Result = Tuple{Float64, Belief, Float64}
function results(m::MetaMDP, b::Belief, c::Int)::Vector{Result}
    is_terminal(b) && error("Belief is terminal.")
    if c == ⊥_ACTION
        return [(1., terminate(b), term_reward(b))]
    end
    r = -cost(m, b, c)
    [(p, update(m, b, c, x), r)
     for (p, x) in discretized_randn(m.quantization)]
end

# %% ==================== Features ====================

function expect_max_dist(d::Distribution, constant::Float64)
    p_improve = 1 - cdf(d, constant)
    p_improve < 1e-10 && return constant
    (1 - p_improve) * constant + p_improve * mean(Truncated(d, constant, Inf))
end

function voi1(m::MetaMDP, b::Belief, c::Int)
    μ = getfield.(b, :μ)
    cv = maximum(μ[i] for i in 1:length(μ) if i != c)
    μ_dist, = posterior(m, b[c])
    expect_max_dist(μ_dist, cv) - maximum(μ)
end

function voi_action(b::Belief, c::Int)
    μ = getfield.(b, :μ)
    cv = maximum(μ[i] for i in 1:length(μ) if i != c)
    expect_max_dist(b[c], cv) - maximum(μ)
end

using Cuba
function vpi(b)
    μ, σ = getfield.(b, :μ), getfield.(b, :σ)
    low, high = μ .- (5 .* σ), μ .+ (5 .* σ)
    mult = prod(high - low)
    g(x, v) = begin
        x .= low .+ (high-low) .* x
        v .= maximum(x; dims=1) .* prod(pdf.(b, x); dims=1) .* mult
    end
     # hcubature(g, zeros(3), ones(3), atol=1e-4);
     cuhre(g, 3, nvec=1000).integral[1] - maximum(μ)
end


# update(INITIAL, 1, true)
# results(INITIAL, 1)
# term_reward(INITIAL)
#
# # %% ==================== Solution ====================
# function Q(b::Belief, action::Int)::Float64
#     sum(p * (r + V(s1)) for (p, s1, r) in results(state, action))
# end
#
# function _V(b::Belief)::Float64
#     state == ⊥_STATE && return 0
#     sum(sum.(state.arms)) > 30 && return term_reward(state)
#     maximum(Q(state, a) for a in ACTIONS)
# end
#
# policy(b::Belief) = ACTIONS[argmax([Q(state, a) for a in ACTIONS])]
#
# # Cache state values
# struct ValueFunction
#     cache::Dict{UInt64, Float64}
# end
# ValueFunction() = ValueFunction(Dict{UInt64, Float64}())
#
# function symmetry_breaking_hash(s::Belief)
#     key = UInt64(0)
#     for i in 1:length(s.arms)
#         key += (hash(s.arms[i]) << 3(i == s.focused))
#     end
#     key
# end
#
# function (V::ValueFunction)(s::Belief)::Float64
#     key = symmetry_breaking_hash(s)
#     haskey(V.cache, key) && return V.cache[key]
#     v = _V(s)
#     V.cache[key] = v
#     return v
# end
# V = ValueFunction()
#
# # %% ==================== Features ====================
# using Distributions
#
# function voi_action(state, action)
#     vals = [p_win(arm) for arm in state.arms]
#     best, second = sortperm(-vals)
#     competing_val = action == best ? vals[second] : vals[best]
#     a, b = state.arms[action]
#     expected_max_beta_constant(a, b, competing_val)
# end
#
# function expected_max_beta_constant(a, b, c)
#     x = 0:0.001:1
#     px = pdf(Beta(a, b), x) ./ length(x)
#     sum(px .* max.(x, c))
# end
