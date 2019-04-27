include("new_metabandits.jl")
# %% ====================  ====================
m = MetaMDP(sample_cost=0.1)
b = Belief(m)
display("")
println(b)
println(voi1(m, b, 1))
println(voi_action(m, b, 1))
println(vpi(m, b))
# %% ====================  ====================
m = MetaMDP(sample_cost=.01, max_obs=30)
b = Belief(m)
V = ValueFunction(m)
@time println(V(b))
# %% ====================  ====================
m = MetaMDP(sample_cost=.01, max_obs=30)
pol = OptimalPolicy(m)

function expectation(f, n)
    acc = 0.
    for i in 1:n
        acc += f()
    end
    acc / n
end
# %% ====================  ====================
ev = expectation(100000) do
    rollout(pol).reward
end
V(Belief(m)) - ev

# %% ====================  ====================
m = MetaMDP(sample_cost=.001, max_obs=5)
pol = BMPSPolicy2(m, [0., 0.9, 0.1, 0.])
@time expectation(1000) do
    rollout(pol).reward
end |> println

using Profile
@profile rollout(pol);
