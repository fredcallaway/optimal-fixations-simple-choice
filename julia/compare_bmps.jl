using Distributed
addprocs()

using Serialization
@everywhere include("blinkered.jl")
include("optimize.jl")

results = "results/2019-06-01T11-36-33"
blinkered = open(deserialize, "$results/blinkered_policy.jls").policy
m = blinkered.m
opt = optimize(m)

open("$results/bmps_opt", "w+") do f
    serialize(f, opt)
end

θ = opt.θ_i[argmax(opt.r_i)]
θ1 = opt.θ1

function expected_reward(policy, nr=1000)
    reward = @distributed (+) for i in 1:nr
        rollout(policy, max_steps=200).reward
    end
    reward / nr
end

expected_reward(Policy(m, θ), 10000)
expected_reward(Policy(m, θ1), 10000)
expected_reward(Blinkered(m), 10000)

open("$results/bmps_policy", "w+") do f
    serialize(f, Policy(m, θ))
end
