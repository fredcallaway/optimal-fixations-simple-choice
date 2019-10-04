using Distributed
include("results.jl")
@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("params.jl")
end

# results = get_result("results/pseudo_3_epsilon/2019-09-24T10-51-38-6fl/")

# %% ====================  ====================
results = get_result("results/pseudo_4_top/2019-09-25T00-49-00-BoJ/")


function reoptimize(prm::Params; N=16)
    policies = asyncmap(1:N) do i
        m = MetaMDP(prm)
        optimize_bmps(m; α=prm.α)
    end
    save(results, :reopt, policies)
end

reoptimize(load(results, :mle_101))
