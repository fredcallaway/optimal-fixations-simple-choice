using Distributions
using Memoize
import Random
using Parameters
import Base
using Printf
# using StaticArrays

@with_kw struct MetaMDP
    n_arm::Int = 3
    sample_cost::Float64 = 0.001
    switch_cost::Float64 = 1
end
# n_arm(m::MetaMDP{N}) where {N} = N
# MetaMDP(;kws...) = MetaMDP{3}(;kws...)  # 3 arms by default
actions(m) = 0:m.n_arm

struct Belief
    value::Vector{Tuple{Int, Int}}
    focused::Int
end


Belief(m::MetaMDP) = begin
    value = [(1,1) for _ in 1:m.n_arm]
    Belief(value, 0)
end

Base.:(==)(b1::Belief, b2::Belief) = b1.focused == b2.focused && b1.value == b2.value
Base.hash(b::Belief) = hash(b.focused, hash(b.value))
Base.getindex(b::Belief, idx) = b.value[idx]
Base.length(b::Belief) = length(b.value)
Base.iterate(b::Belief) = iterate(b.value)
Base.iterate(b::Belief, i) = iterate(b.value, i)

function Base.show(io::IO, b::Belief)
    print(io, "[ ")
    value = map(1:length(b.value)) do i
        h, t = b.value[i]
        i == b.focused ? @sprintf("<%02d %02d>", h, t) : @sprintf(" %02d %02d ", h, t)
    end
    print(io, join(value, " "))
    # for a in b.arms
        # print(io, a[1], " ", a[2])
    # end
    print(io, " ]")
end

@isdefined(TERM_ACTION) || const TERM_ACTION = 0

function update(b::Belief, arm::Int, heads::Bool)::Belief
    value = copy(b.value)
    a, b = value[arm]
    value[arm] = heads ? (a+1, b) : (a, b+1)
    Belief(value, arm)
end

function cost(m::MetaMDP, b::Belief, c::Int)::Float64
    m.sample_cost * (c != b.focused ? m.switch_cost : 1)
end

p_heads(arm) = arm[1] / (arm[1] + arm[2])

term_reward(b::Belief)::Float64 = maximum(p_heads.(b.value))
is_terminal(b::Belief) = b.focused == -1
terminate(b::Belief) = Belief(b.value, -1)

Result = Tuple{Float64, Belief, Float64}
function results(m::MetaMDP, b::Belief, c::Int)::Vector{Result}
    is_terminal(b) && error("Belief is terminal.")
    if c == TERM_ACTION
        return [(1., terminate(b), term_reward(b))]
    end
    p1 = p_heads(b.value[c])
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
    vals = [p_heads(arm) for arm in b.value]
    best, second = sortperm(-vals)
    competing_val = c == best ? vals[second] : vals[best]
    h, t = b.value[c]
    expected_max_beta_constant(h, t, competing_val) - term_reward(b)
end

function vpi(m::MetaMDP, b; n_sample=10000)
    dists = [Beta(a...) for a in b.value]
    x = rand.(dists, 10000)
    mean(max.(x[1], x[2], x[3])) - term_reward(b)
end

function expected_max_beta_constant(h, t, k)
    x = 0:0.001:1
    px = pdf(Beta(h, t), x) ./ length(x)
    sum(px .* max.(x, k))
end

# update(INITIAL, 1, true)
# results(INITIAL, 1)
# term_reward(INITIAL)
#
# # %% ==================== Solution ====================
# function Q(b::Belief, c::Int)::Float64
#     sum(p * (r + V(s1)) for (p, s1, r) in results(b, c))
# end
#
# function _V(b::Belief)::Float64
#     b == TERM_STATE && return 0
#     sum(sum.(b.value)) > 30 && return term_reward(b)
#     maximum(Q(b, a) for a in ACTIONS)
# end
#
# policy(b::Belief) = ACTIONS[argmax([Q(b, a) for a in ACTIONS])]
#
# # Cache b values
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
# function voi_action(b, c)
#     vals = [p_win(arm) for arm in b.value]
#     best, second = sortperm(-vals)
#     competing_val = c == best ? vals[second] : vals[best]
#     a, b = b.value[c]
#     expected_max_beta_constant(a, b, competing_val)
# end
#
# function expected_max_beta_constant(a, b, c)
#     x = 0:0.001:1
#     px = pdf(Beta(a, b), x) ./ length(x)
#     sum(px .* max.(x, c))
# end
