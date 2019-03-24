include("bernoulli_metabandits.jl")
# display("")
s = State([(10,10), (10,4), (3, 1)], 1)
Q(s, 1)
@time println(V(INITIAL));

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


[2 1  1 1  (3 4)]

roll()


# %% ====================  ====================
D = Dict(s=>1)
s1 = State([(10,10), (3, 1), (10,4)], 1)
haskey(D, s1)
hash(s1) == hash(s)
# _V[INITIAL]
# %% ====================  ====================
@code_warntype Q(INITIAL, 1)
@code_warntype V(INITIAL)
# %% ====================  ====================
State_

println(1)
using Profile
s = ([(10,10), (10,10)], 1)
Profile.clear()
@profile Q(INITIAL, 1);
Profile.print()

0.7058041883646557
  0.241321 seconds (2.00 M allocations: 59.743 MiB, 5.63% gc time)
