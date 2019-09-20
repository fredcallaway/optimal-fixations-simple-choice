using Distributed
include("results.jl")
@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("params.jl")
end
include("pseudo_likelihood.jl")

run_name = "fit_pseudo_3_reopt"
prm = open(deserialize, "tmp/fit_pseudo_3_model_mle")
m = MetaMDP(prm)
policy = optimize_bmps(m, α=prm.α)[1]
save(Results(run_name), :policy, policy)


name = "fit_pseudo_4_reopt"
policy = open(deserialize, "tmp/fit_pseudo_4_model_mle")

policies = map(get_results("fit_pseudo_3_reopt")) do res
    policy = load(res, :policy)
    typeof(policy)
end


cfp = Metric(chosen_fix_proportion, 2)
h = cfp.(trials)
X = map([3,4]) do n
    map(get_results("fit_pseudo_$(n)_reopt")) do res
        policy = load(res, :policy)
        sim = simulate_experiment(policy; n_repeat=1)
        m = cfp.(sim)
        mean(h), mean(m), mean(h .== m)
    end
end

using SplitApplyCombine
X = combinedims(XXX)
X = combinedims(collect.(X))

mean(X[3, :, :]; dims=1)

# new_res = Results(name)
# new_pol = optimize_bmps(policy.m, α=policy.α)[1]
# save(new_res, :policy, new_pol)


# name = ARGS[1]
# i = length(ARGS) > 1 ? parse(Int, ARGS[1]) : 1
# res = get_results(name)[i]
# policy = load(res, :policy)

# new_res = Results(join([name, i, "reopt"], "_"))
# new_pol = optimize_bmps(policy.m, α=policy.α)[1]
# save(new_res, :policy, new_pol)
