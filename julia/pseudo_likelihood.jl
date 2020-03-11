using Distributed
using Optim

include("human.jl")
include("binning.jl")
include("simulations.jl")
include("features.jl")
include("metrics.jl")

function get_metrics(metrics, policies, prior, v, N, parallel)
    apply_metrics = juxt(metrics...)
    # map(1:N) do i
    if parallel
        @distributed vcat for i in 1:N
            policy = policies[1 + i % length(policies)]
            sim = simulate(policy, prior, v)  #  .+ prior.σ_rating .* randn(length(v))
            apply_metrics(sim)
        end
    else
        map(1:N) do i
            policy = policies[1 + i % length(policies)]
            sim = simulate(policy, prior, v)  #  .+ prior.σ_rating .* randn(length(v))
            apply_metrics(sim)
        end
    end
end

function make_histogram(metrics, policies, prior, v, N, parallel)
    histogram_size = Tuple(length(m.bins) for m in metrics)
    apply_metrics = juxt(metrics...)
    L = zeros(histogram_size...)
    for m in get_metrics(metrics, policies, prior, v, N, parallel)
        if any(ismissing(x) for x in m)
            println("-----------------------")
            println(m)
            error("Missing index")
        end
        L[m...] += 1
    end
    L ./ sum(L)
end


# %% ====================  ====================

@memoize function make_metrics(trials)
    n_item = length(trials[1].value)
    hb = LIKELIHOOD_PARAMS.hist_bins
    metrics = [
        Metric(total_fix_time, hb, trials),
        Metric(n_fix, Binning([0; 2:hb; Inf])),
        Metric(t->t.choice, Binning(1:n_item+1)),
    ]
    for i in 1:(n_item-1)
        push!(metrics, Metric(t->propfix(t)[i], hb, trials))
    end
    return metrics
end

function raw_likelihood(trials, metrics, policies, prior, n_sim_hist)
    parallel = false
    vs = unique(trials.value);
    histograms = map(vs) do v
        yield()  # for Toucher
        v => make_histogram(metrics, policies, prior, v, n_sim_hist, parallel)
    end |> Dict

    apply_metrics = juxt(metrics...)
    function likelihood(t)
        L = histograms[t.value]
        L[apply_metrics(t)...]
    end
    likelihood.(trials), histograms;
end

function likelihood(policies, β_μ; fit_ε, max_ε, n_sim_hist, hist_bins, test_fold, fold)
    all_trials = map(sort_value, load_dataset(policies[1].m.n_arm))
    metrics = make_metrics(all_trials)
    trials = get_fold(all_trials, test_fold, :test)
    # trials = getfield(split_trials, fold)
    prior = make_prior(trials, β_μ)
    parallel = false

    histogram_size = Tuple(length(m.bins) for m in metrics)
    p_rand = 1 / prod(histogram_size)
    baseline = log(p_rand) * length(trials)

    raw, _ = raw_likelihood(trials, metrics, policies, prior, n_sim_hist)
    loglike(ε) = sum(@. log(ε * p_rand + (1 - ε) * raw))

    if fit_ε
        ε = Optim.optimize(ε->-loglike(ε), 0, max_ε).minimizer
    else  # use ε equivalent to add-one smoothing
        ε = prod(histogram_size) / (n_sim_hist + prod(histogram_size))
    end

    loglike(ε), ε, baseline
end
