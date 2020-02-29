include("fit_base.jl")
include("meta_mdp.jl")
include("bmps.jl")
include("pseudo_likelihood.jl")
include("pseudo_base.jl")

function compute_likelihood(job::Int)
    datasets = [build_dataset("two", -1), build_dataset("three", -1)]
    both_policies = deserialize("$BASE_DIR/policies/$job")
    results = map(0.5:0.05:1) do β_μ
        losses = map(1:2) do item_idx
            policies = both_policies[item_idx]
            logp, ε, baseline = likelihood(policies, β_μ; LIKELIHOOD_PARAMS..., fold=:train)
            logp / baseline
        end
        prm = get_prm(job)
        pol = both_policies[1][1];
        @assert pol.α == prm.α && pol.m.σ_obs == prm.σ_obs
        prm = (prm..., β_μ=β_μ)
        prm, losses
    end
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    job = parse(Int, ARGS[1])
    do_job(compute_likelihood, "likelihood", job)
end