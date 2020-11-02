using Distributed
using StatsPlots
using Serialization
using Plots.Measures
using SplitApplyCombine

pwd()
pyplot(label="")
include("meta_mdp.jl")
include("bmps.jl")

policies = deserialize("results/main14/test_policies/joint-false/1");
pol2 = policies[1][1]
pol3 = policies[2][1]

# %% ====================  ====================
function simulate(policy, value; max_steps=1000)
    s = State(policy.m, value)
    bs = Belief[]; cs = Int[]
    roll = rollout(policy, state=s; max_steps=max_steps) do b, c
        push!(bs, copy(b))
        push!(cs, c)
    end
    bs, cs
end


function plot_accumulation(v)
    bs, cs = simulate(pol2, [1., 3.])
    μ = combinedims(getfield.(bs, :μ))'
    λ = combinedims(getfield.(bs, :λ))'
    σ = λ .^ -0.5
    plot(μ, ribbon=σ)
end

plot_accumulation(zeros(2))
# %% ====================  ====================

pol = pol3
init = Belief(pol.m)
init.μ[2:3] = [-1, 1]

b = copy(init)
μs = -2:0.01:2
v1, v2, v3 = map(μs) do μ1
    b.μ[1] = μ1
    # voc(pol, b)[1]
    voi1(b, 1), voi_action(b, 1), vpi(b)
end |> invert
plot(μs, [v1 v2 v3])

# %% ====================  ====================
b = Belief(pol.m)
λs = 1:.01:5
b = copy(init)
v1, v2, v3 = map(λs) do λ1
    b.λ[1] = λ1
    # voc(pol, b)[1]
    voi1(b, 1), voi_action(b, 1), vpi(b)
end |> invert
plot(λs, [v1 v2 v3])
