include("bernoulli_metabandits.jl")


m = MetaMDP(switch_cost=s)
b = Belief(m)
V = ValueFunction(m)
V(b)


# %% ====================  ====================
include("optimize_bmps.jl")
policies, μ, sem, hist, converged = ucb(m; N=1000, n_iter=2^17, n_roll=64, n_init=4, β=3.)


# %% ====================  ====================
policy = BMPSPolicy(m, [0, 1, 0, 1])
@time rollout(policy)

# %% ====================  ====================
include("optimize_bmps.jl")

sample_cost, switch_cost = (0.00012340980408667956, 1)
m = MetaMDP(sample_cost=sample_cost, switch_cost=switch_cost, max_obs=10)
V = ValueFunction(m)
opt_val = V(Belief(m))
bmps_pol, bmps_val = optimize_bmps(m; N=8)

id = round(Int, rand() * 1e8)
file = "bernoulli/$id"
open(file, "w+") do f
    serialize(f, (
        m=m,
        bmps_val=bmps_val,
        opt_val=opt_val,
        time_stamp=now()
    ))
end