# State = Vector{Tuple{Int, Int}}
INITIAL = ([(1,1), (1,1)], 1)
# const TERM_STATE = ([(0, 0)], 0)

State_ = typeof(INITIAL)
Result = Tuple{Float64, State_, Float64}

COST = 0.001
ACTIONS = [0,1,2]

p_win(arm) = arm[1] / (arm[1] + arm[2])

function update(state::State_, arm::Int, result::Bool)
    arms, focused = state
    arms = copy(arms)
    a, b = arms[arm]
    arms[arm] = result ? (a+1, b) : (a, b+1)
    (arms, arm)
end

function results(state::State_, action::Int)::Vector{Result}
    if action == 0
        return [(1., TERM_STATE, term_reward(state))]
    end
    a, b = state[1][action]
    p1 = p_win(state[1][action])
    p0 = 1 - p1
    [(p0, update(state, action, false), -COST),
     (p1, update(state, action, true), -COST)]
end

term_reward(state::State_) = maximum(p_win.(state[1]))

update(INITIAL, 1, true)
results(INITIAL, 1)
term_reward(INITIAL)

# %% ====================  ====================
function Q(state::State_, action::Int64)::Float64
    sum(p * (r + V(s1)) for (p, s1, r) in results(state, action))
end

_V = Dict{State_, Float64}()
function V(state::State_)::Float64
    haskey(_V, state) && return _V[state]
    state == TERM_STATE && return 0
    sum(sum.(state[1])) > 40 && return term_reward(state)
    v = maximum(Q(state, a) for a in ACTIONS)
    _V[state] = v
    v
end
