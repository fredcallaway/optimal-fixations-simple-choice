include("results.jl")
@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
end

name = "sep15_200_f1"
i = 1
res = get_results(name)[i]
policy = load(res, :policy)

new_res = Results(join([name, i, "reopt"], "_"))
new_pol = optimize_bmps(policy.m, α=policy.α)[1]
save(new_res, :policy, new_pol)
