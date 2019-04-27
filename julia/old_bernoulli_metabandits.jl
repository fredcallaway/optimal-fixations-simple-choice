import Base.copy
using SplitApplyCombine
COST = 0.0001
SWITCH_COST = 4
ACTIONS = [0,1,2,3]
N_ARM = 3
TERM = 0


struct State
    arms::Vector{Tuple{Int, Int}}
    focused::Int64
end
Base.:(==)(s1::State, s2::State) = s1.focused == s2.focused && s1.arms == s2.arms

@isdefined(INITIAL) || const INITIAL = State([(1,1), (1,1), (1,1)], 0)
@isdefined(TERM_STATE ) || const TERM_STATE = State([(0, 0)], 0)

p_win(arm) = arm[1] / (arm[1] + arm[2])

function update(state::State, arm::Int, heads::Bool)::State
    arms = copy(state.arms)
    a, b = arms[arm]
    arms[arm] = heads ? (a+1, b) : (a, b+1)
    State(arms, arm)
end

function cost(state::State, arm::Int)::Float64
    COST * (arm != state.focused ? SWITCH_COST : 1)
end

Result = Tuple{Float64, State, Float64}
function results(state::State, action::Int)::Vector{Result}
    if action == TERM
        return [(1., TERM_STATE, term_reward(state))]
    end
    a, b = state.arms[action]
    p1 = p_win(state.arms[action])
    p0 = 1 - p1
    [(p0, update(state, action, false), -cost(state, action)),
     (p1, update(state, action, true), -cost(state, action))]
end

term_reward(state::State)::Float64 = maximum(p_win.(state.arms))

# update(INITIAL, 1, true)
# results(INITIAL, 1)
# term_reward(INITIAL)

# %% ==================== Solution ====================
function Q(state::State, action::Int)::Float64
    sum(p * (r + V(s1)) for (p, s1, r) in results(state, action))
end

function _V(state::State)::Float64
    state == TERM_STATE && return 0
    sum(sum.(state.arms)) > 30 && return term_reward(state)
    maximum(Q(state, a) for a in ACTIONS)
end

policy(state::State) = ACTIONS[argmax([Q(state, a) for a in ACTIONS])]

# Cache state values
struct ValueFunction
    cache::Dict{UInt64, Float64}
end
ValueFunction() = ValueFunction(Dict{UInt64, Float64}())

function symmetry_breaking_hash(s::State)
    key = UInt64(0)
    for i in 1:length(s.arms)
        key += (hash(s.arms[i]) << 3(i == s.focused))
    end
    key
end

function (V::ValueFunction)(s::State)::Float64
    key = symmetry_breaking_hash(s)
    haskey(V.cache, key) && return V.cache[key]
    v = _V(s)
    V.cache[key] = v
    return v
end

# %% ==================== Features ====================
using Distributions

function voi1(state, action)
    sum(p * (term_reward(s1) + r ) for (p,s1,r) in results(state, action)) - term_reward(state)
end

function voi_action(state, action)
    vals = [p_win(arm) for arm in state.arms]
    best, second = sortperm(-vals)
    competing_val = action == best ? vals[second] : vals[best]
    a, b = state.arms[action]
    expected_max_beta_constant(a, b, competing_val) - term_reward(state)
end

function vpi(state; n_sample=10000)
    dists = [Beta(a...) for a in state.arms]
    x = rand.(dists, 10000)
    mean(max.(x[1], x[2], x[3])) - term_reward(state)
end

function expected_max_beta_constant(a, b, c)
    x = 0:0.001:1
    px = pdf(Beta(a, b), x) ./ length(x)
    sum(px .* max.(x, c))
end

function features(s::State)
    vpi_ = vpi(s)
    phi(c) = [
        -1,
        voi1(s, c),
        voi_action(s, c),
        vpi_,
    ]
    combinedims([phi(c) for c in 1:N_ARM])
end

# %% ==================== Policy ====================
noisy(x, ε=1e-10) = x .+ ε .* rand(length(x))

struct Policy
    θ::Vector{Float64}
end

function voc(θ, s)
    x = (pol.θ' * features(s))'
    x .-= [cost(s, c) for c in 1:N_ARM]
    x
end

(pol::Policy)(s::State) = begin
    v, c = findmax(noisy(voc(pol.θ, s)))
    v <= 0 ? TERM : c
end
