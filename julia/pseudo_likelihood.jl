using Distributed
using Optim

include("meta_mdp.jl")
include("bmps.jl")
include("optimize_bmps.jl")
include("human.jl")
include("binning.jl")
include("simulations.jl")
include("features.jl")
include("params.jl")
include("metrics.jl")


const N_SIM_HIST = 10_000
const SAMPLE_TIME = 100
const MAX_STEPS = 200  # 20 seconds

function sim_one(policy, prm, v)
    sim = simulate(policy, (v .- prm.μ) ./ prm.σ; max_steps=MAX_STEPS)
    fixs, fix_times = parse_fixations(sim.samples, SAMPLE_TIME)
    (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times)
end

function get_metrics(metrics, policies, prm, v, N, parallel)
    apply_metrics = juxt(metrics...)
    # map(1:N) do i
    if parallel
        @distributed vcat for i in 1:N
            policy = policies[1 + i % length(policies)]
            sim = sim_one(policy, prm, v)  #  .+ prm.σ_rating .* randn(length(v))
            apply_metrics(sim)
        end
    else
        map(1:N) do i
            policy = policies[1 + i % length(policies)]
            sim = sim_one(policy, prm, v)  #  .+ prm.σ_rating .* randn(length(v))
            apply_metrics(sim)
        end
    end
end

function likelihood_matrix(metrics, policies, prm, v, N, parallel)
    histogram_size = Tuple(length(m.bins) for m in metrics)
    apply_metrics = juxt(metrics...)
    L = zeros(histogram_size...)
    for m in get_metrics(metrics, policies, prm, v, N, parallel)
        if any(ismissing(x) for x in m)
            println("-----------------------")
            println(m)
            error("Missing index")
        end
        L[m...] += 1
    end
    L ./ sum(L)
end

function total_likelihood(policies, prm, trials; fit_ε, max_ε, metrics, n_sim_hist=N_SIM_HIST, parallel=true)
    vs = unique(sort(t.value) for t in trials);
    sort!(vs, by=std)  # fastest rank_trials last for parallel efficiency
    out = asyncmap(vs, ntasks=5) do v
        likelihood_matrix(metrics, policies, prm, v, n_sim_hist, parallel)
    end
    likelihoods = Dict(zip(vs, out))

    apply_metrics = juxt(metrics...)
    histogram_size = Tuple(length(m.bins) for m in metrics)
    p_rand = 1 / prod(histogram_size)
    baseline = log(p_rand) * length(trials)

    function likelihood(t)
        L = likelihoods[sort(t.value)]
        L[apply_metrics(t)...]
    end

    X = likelihood.(trials);

    f(ε) = sum(@. log(ε * p_rand + (1 - ε) * X))

    if fit_ε
        ε = Optim.optimize(ε->-f(ε), 0, max_ε).minimizer
    else
        ε = prod(histogram_size) / (n_sim_hist + prod(histogram_size))
    end
    f(ε), ε, baseline, likelihoods
end
