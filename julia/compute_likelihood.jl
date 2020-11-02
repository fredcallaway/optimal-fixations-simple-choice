include("fit_base.jl")
include("pseudo_likelihood.jl")
using Random

function compute_likelihood(job::Int, prior_i)
    prm = get_prm(job)
    if :p_stop in keys(SPACE.dims)
        # unoptimized random policy
        all_policies = [[Rando(MetaMDP(n, prm), prm.p_switch, prm.p_stop)] for n in 2:3]
    else
        all_policies = deserialize("$BASE_DIR/policies/$job")
        # check that parameters are correct
        pol = all_policies[1][1];
        @assert pol.α == prm.α && pol.m.σ_obs == prm.σ_obs
    end


    β_μs = FIT_PRIOR ? collect(range(0, 1, length=GRID_SIZE)) : [1.0]
    if SEARCH_STRATEGY == :sobol
        noise = rand(MersenneTwister(job)) # make it reproducible
        β_μs[2:end-1] .+= (1 / (GRID_SIZE-1)) * 2(noise - 0.5)
    end
    β_μ = β_μs[prior_i]

    hists = map(all_policies) do policies
        make_histograms(policies, β_μ, LIKELIHOOD_PARAMS.n_sim_hist)
        # @time make_histograms(policies, β_μ, 100);
        # logp, ε, baseline = likelihood(policies, β_μ; LIKELIHOOD_PARAMS..., fold=:train)
    end
    prm = (prm..., β_μ=β_μ)
    prm, hists
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    # job = parse(Int, ARGS[1])
    job = eval(Meta.parse(ARGS[1]))
    prior_i = parse(Int, ARGS[2])
    do_job(compute_likelihood, "likelihood/$prior_i", job, prior_i)
end


# h = 100_000
# dpch = 4.608 / 96
# dpch * h

# all histograms: 15M
# one histogram: 144K
# one set of 50k simulations: 4 MB (3900K)
# 182 unique values
# 4 * 182 = 728 MB
# 3900 / 144 = 27
# full simulations: 15 * 27 = 405 M