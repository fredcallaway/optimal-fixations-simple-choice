include("model.jl")

using Profile
using BenchmarkTools

function main()
    pomdp = Pomdp()
    b = Belief(pomdp)
    s = State(pomdp)
    # println(s)
    # println(b)
    # println(string("r = ", step!(pomdp, b, s, 1)))
    # println(b)
    
    # b.mu[1] = 2
    # println(voi1(b, 1, pomdp.obs_sigma ^ -2))
    # println(voi_action(b, 1))
    # println(vpi(b))
    policy = bmps_policy([0., 0, 1, 0])
    rollout(pomdp, policy)
    @time [rollout(pomdp, policy) for i in 1:10]
    @profile [rollout(pomdp, policy) for i in 1:10]
    Profile.print(noisefloor=2)

    
end
# main()





