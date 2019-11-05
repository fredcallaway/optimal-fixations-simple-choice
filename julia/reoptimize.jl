using Distributed
include("results.jl")
@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("params.jl")
end

results = get_result(ARGS[1])
# results = get_result("results/both_items_fixed_parallel_post/2019-10-28T12-31-17-5Zc/")

function reoptimize(prm::Params; N=16)
    reopt = map(1:2) do i
        policies = asyncmap(1:N) do j
            m = MetaMDP(i+1, prm)
            optimize_bmps(m; α=prm.α)
        end
        (policies=policies,)
    end
    save(results, :reopt, reopt)
end


# prm = load(results, :mle)
prm = load(results, :mle)
reoptimize(prm)
