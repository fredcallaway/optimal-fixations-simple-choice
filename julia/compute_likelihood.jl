include("fit_base.jl")
include("pseudo_likelihood.jl")

function compute_likelihood(job::Int)
    all_policies = deserialize("$BASE_DIR/policies/$job")
    # sanity check
    prm = get_prm(job)
    pol = all_policies[1][1];
    @assert pol.α == prm.α && pol.m.σ_obs == prm.σ_obs

    β_μs = FIT_PRIOR ? range(0, 1, length=GRID_SIZE) : [1.0]
    results = map(β_μs) do β_μ
        losses = map(all_policies) do policies
            logp, ε, baseline = likelihood(policies, β_μ; LIKELIHOOD_PARAMS..., fold=:train)
            -logp
        end
        prm = (prm..., β_μ=β_μ)
        prm, losses
    end
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    job = parse(Int, ARGS[1])
    do_job(compute_likelihood, "likelihood", job)
end