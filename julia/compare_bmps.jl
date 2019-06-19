using Distributed
addprocs()

using Serialization
@everywhere include("blinkered.jl")


results = "results/2019-06-01T11-36-33"
blinkered = open(deserialize, "$results/blinkered_policy.jls").policy
m = blinkered.m

# %% ====================  ====================
include("optimize.jl")
opt = optimize(m)
open("$results/bmps_opt", "w+") do f
    serialize(f, opt)
end

θ = opt.θ_i[argmax(opt.r_i)]
θ1 = opt.θ1

open("$results/bmps_policy", "w+") do f
    serialize(f, Policy(m, θ))
end
# %% ====================  ====================

bmps_policy = open(deserialize, "$results/bmps_policy")

function expected_reward(policy, nr=1000)
    reward = @distributed (+) for i in 1:nr
        rollout(policy, max_steps=200).reward
    end
    reward / nr
end

results = Dict(
    :bmps => expected_reward(bmps_policy, 10000),
    :soft_blinkered => expected_reward(blinkered, 10000),
    :hard_blinkered => expected_reward(Blinkered(m), 10000)
)

