# %%
include("inference_helpers.jl")
include("blinkered.jl")


# %% ==================== Simulate data ====================

true_prm = Params(
    α = 52.73019415710551,
    obs_sigma = 13.985483892291345,
    sample_cost = 0.0009382165229567387,
    switch_cost = 54.48621024645527,
    μ = 4.517473715259105,
    σ = 0.6309531499105546,
)

value(prm::Params, v::Vector{Float64}) = (v .- prm.µ) ./ prm.σ
function simulate(prm, v)::Datum
    policy = SoftBlinkered(prm)
    cs = Int[]
    s = State(policy.m, value(prm, v))
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c); max_steps=1000)
    Datum(v, cs[1:end-1], roll.choice)
end

data = simulate.(true_prm, trials.value[1:10])

true_mdp = MetaMDP(obs_sigma=10, switch_cost=8, sample_cost=0.001)
true_α = 100.
true_policy = SoftBlinkered(true_mdp, true_α)

sim_data = map(1:100) do i
    simulate(true_policy, randn(3))
end

# %% ====================  ====================

# function plogp(policy, ε)
#     @distributed (+) for d in sim_data
#         logp(policy, ε, d.value, d.samples, d.choice, 10000)
#     end
# end
# @time plogp(true_policy, 0.05);

function logp(policy)
    mapreduce(+, sim_data) do d
        logp(policy, d.value, d.samples, d.choice, 10000)
    end
end

d = sim_data[1]
@time logp(true_policy, d.value, d.samples, d.choice, 100)
# @time logp(true_policy, 0.05);

# %% ====================  ====================

alt_mdp = MetaMDP(obs_sigma=1, switch_cost=8, sample_cost=0.001)
logp(true_policy, d.value, d.samples, d.choice, 1000)
logp(SoftBlinkered(alt_mdp, 100.), d.value, d.samples, d.choice, 1000)


# %% ==================== Recover parameters? ====================
@everywhere rescale(x, low, high) = low + x * (high-low)

@everywhere MetaMDP(x::Vector{Float64}) = MetaMDP(
    3,
    rescale(x[1], 1, 5),
    10 ^ rescale(x[2], -5, -3),
    rescale(x[3], 1, 10)
)

true_logp = plogp(true_policy, 0.05)
candidates = [MetaGreedy(MetaMDP(rand(3))) for i in 1:960]
@time logps = pmap(candidates) do pol
    logp(pol, 0.05)
end;
best = candidates[argmax(logps)].m
cv_logp = plogp(MetaGreedy(best), 0.05)

# %% ====================  ====================
using Printf
display("")
println(true_mdp)
@printf "%.1f vs. %.1f vs. %.1f\n" true_logp maximum(logps) cv_logp
@printf "obs_sigma    %.1f vs. %.1f\n" true_mdp.obs_sigma true_mdp.obs_sigma
@printf "sample_cost  %.1f vs. %.1f\n" log(true_mdp.sample_cost) log(best.sample_cost)
@printf "switch_cost  %.1f vs. %.1f\n" true_mdp.switch_cost best.switch_cost

