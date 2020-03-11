include("fit_base.jl")
include("ucb_bmps.jl")


function compute_policies(n_item::Int, prm::NamedTuple; kws...)
    if hasproperty(prm, :p_switch)
        kws = (kws..., p_switch=prm.p_switch)
    end
    m = MetaMDP(n_item, prm.σ_obs, prm.sample_cost, prm.switch_cost)
    policies, μ, sem = ucb(m; α=prm.α, UCB_PARAMS..., kws...)
    best = partialsortperm(-μ, 1:UCB_PARAMS.n_top)
    return policies[best]
end

function compute_policies(job::Int)
    prm = get_prm(job)
    map(2:3) do n_item
        compute_policies(n_item, prm)
    end
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    job = parse(Int, ARGS[1])
    do_job(compute_policies, "policies", job)
end
