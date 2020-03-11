using Serialization
using SplitApplyCombine
using Glob

@everywhere include("bernoulli_meta_mdp.jl")
@everywhere using Serialization

# %% ==================== Horizon hitting ====================
files = filter(glob("results/bernoulli/*")) do f
    !endswith(f, "csv")
end

hit_horizon_rate = pmap(files) do f
    res = deserialize(f)
    policies, μ, sem = res.ucb
    best = partialsortperm(-μ, 1:30)
    300 \ mapreduce(+, policies[best]) do pol
        mapreduce(+, 1:10) do i
            rollout(pol).steps == 76
        end
    end
end


# %% ==================== Build CSV ====================

xs = []
for (f, hhr) in zip(files, hit_horizon_rate)
    res = deserialize(f);
    x = (sample_cost=res.m.sample_cost, switch_cost=res.m.switch_cost)
    push!(xs, (x..., hhr=hhr, agent="optimal", value=res.opt_val))
    for v in res.ucb_vals
        push!(xs, (x..., hhr=hhr, agent="bmps", value=v))
    end
end
using DataFrames
using CSV
DataFrame(xs) |> CSV.write("results/bernoulli/performance.csv")


