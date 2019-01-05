include("model.jl")
using SplitApplyCombine
@timed(rollout(m, policy, max_steps=100))[2]

m = MetaMDP(3, 7, 0.002, 8)
θ = [0.05, 0.8, 0, 0.2]
policy = Policy(m, θ)
s = State(m)
@timed(mean(rollout(m, policy; state=s).steps for i in 1:100))[1:2]

smart = FastPolicy(m, θ)

using Distributed
addprocs(4)
@everywhere include("model.jl")
function test(policy; n=100)
    pmap(1:n) do i
        Random.seed!(i)
        r = rollout(m, policy)
        (r.steps, r.reward)
    end |> invert
end

ts = test(smart; n=10000)
tp = test(policy; n=10000)

mean(ts[2] .- tp[2])

mg = MetaGreedy(m)
@timed(mean(rollout(m, mg; state=s).steps for i in 1:100))[1:2]

using Profile
Profile.init(Int(1e10), 0.01)
@time mean(rollout(m, mg; state=s).steps for i in 1:100)


function foo(π)
    x = Float64[]
    y = Float64[]
    rollout(m, smart) do b, c
        push!(x, vpi(b))
        voc1 = [voi1(b, c) - cost(π.m, b, c) for c in 1:π.m.n_arm]
        push!(y, maximum(voc1))
    end
    x, y
end

x, y = foo(smart)
plot([x y * 10])
hline!([0], c=:black)
