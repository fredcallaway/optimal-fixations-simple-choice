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
# results = get_result("results/pseudo_mu_cv/2019-10-11T14-54-09-Btf")
# results = get_result("results/fit_pseudo_preopt/2019-10-13T10-37-07-KEO/")
# results = get_result("results/fit_pseudo_preopt/2019-10-14T11-43-42-IoI/")
results = get_result(ARGS[1])

function reoptimize(prm::Params; N=16)
    policies = asyncmap(1:N) do i
        m = MetaMDP(prm)
        optimize_bmps(m; α=prm.α)
    end
    save(results, :reopt, policies)
end

reoptimize(load(results, :mle))
