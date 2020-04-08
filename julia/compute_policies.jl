include("fit_base.jl")
include("bmps_ucb.jl")

function compute_policies(n_item::Int, prm::NamedTuple; kws...)
    if hasproperty(prm, :p_switch)
        kws = (kws..., p_switch=prm.p_switch)
    end
    params = merge(UCB_PARAMS, kws)
    m = MetaMDP(n_item, prm)
    policies, μ, sem = ucb_policies(m; α=prm.α, params...)
    best = partialsortperm(-μ, 1:params.n_top)
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
