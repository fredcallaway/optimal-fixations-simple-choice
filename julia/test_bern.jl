include("bernoulli_metabandits.jl")
# display("")
s = State([(10,10), (10,4), (3, 1)], 1)
Q(s, 1)
@time println(V(INITIAL));

# %% ====================  ====================
s = State([(1,1), (1,1), (1,1)], 1)
using Random: rand!
a = 1
voi_action(s, a)
using Memoize
@memoize mem_zeros(shape...) = zeros(shape...)
# %% ====================  ====================

function argmaxes(x)
    r = Set{Int}()
    m = maximum(x)
    for i in eachindex(x)
        if x[i] == m
            push!(r, i)
        end
    end
    r
end


# %% ====================  ====================
p = 1
θ = [0., 1., 0., 0.]
s = INITIAL

possible = argmaxes(voc(θ, s))
p /= length(possible)
for a in possible
    if a == TERM
        v += p * term_reward(s)
    else
        rec(s, p)
    end
end
results(s, a)

function value(θ)
    function rec(s)

    end
end


# %% ====================  ====================
using StatsBase

function roll()
    s = INITIAL
    while s != TERM_STATE
        a = policy(s)
        R = results(s, a)
        println(s, "  ", a, "  ", voi_policy(s))
        p, s, r = sample(R, Weights(getindex.(R, 1)))
    end
end
display("")
roll()

# %% ====================  ====================
function voi_policy(s::State)
    n = length(ACTIONS)
    v = zeros(n)
    for a in 1:n-1
        v[a] = voi_action(s, a) - cost(s, a)
    end
    v[n] = term_reward(s)
    argmax(v) % n
end

s = State([(1, 1), (1, 1), (1, 1)], 2)
voi_policy(s)



# %% ====================  ====================
using Printf

function Base.show(io::IO, s::State)
    print(io, "[ ")
    arms = map(1:length(s.arms)) do i
        a, b = s.arms[i]
        i == s.focused ? @sprintf("<%02d %02d>", a, b) : @sprintf(" %02d %02d ", a, b)
    end
    print(io, join(arms, " "))
    # for a in s.arms
        # print(io, a[1], " ", a[2])
    # end
    print(io, " ]")
end
# %% ====================  ====================
m = MetaMDP(sample_cost=0.001, max_obs=10)
ValueFunction(m)(Belief(m))
expectation


