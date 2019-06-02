include("model.jl")
include("job.jl")

m = MetaMDP(obs_sigma=5)
pol = MetaGreedy(m)
using Statistics
using StatsBase
using Glob
# %% ====================  ====================
# b = Belief([2., 1., 0.], [1., 1., 1.], [1., 1., 1.], 3)

function simulate(policy, value)
    cs = Int[]
    s = State(policy.m, value)
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c); max_steps=1000)
    (samples=cs[1:end-1], choice=roll.choice, value=value)
end

# %% ====================  ====================
m = MetaMDP(obs_sigma=5, switch_cost=1, sample_cost=0.001)
pol = MetaGreedy(m)

x = zeros(3)
for i in 1:1000
    s = simulate(pol, [0., 0.5, 1.]).samples
    x .+= counts(s, 3)
end
round.(x / sum(x), digits=2)
