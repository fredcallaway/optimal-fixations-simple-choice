using Distributed
include("results.jl")
@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("params.jl")
end

results = get_result(ARGS[1])

function reoptimize(prm::Params; N=16)
    policies = asyncmap(1:N) do i
        m = MetaMDP(prm)
        optimize_bmps(m; α=prm.α)
    end
    save(results, :reopt, policies)
end

prm = load(results, :mle)
prm = load(results, :mle_101)
reoptimize(prm)
