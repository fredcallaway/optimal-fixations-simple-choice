@everywhere include("compute_policies.jl")
@everywhere include("pseudo_likelihood.jl")
@everywhere using Optim

# %% --------
best = deserialize("$BASE_DIR/best_parameters/joint-fit")

@everywhere function floss(x)
    any(x .<= 0) && return 25000  # not allowed
    prm = NamedTuple{(:α, :σ_obs, :sample_cost, :switch_cost, :β_μ)}(x)
    cl, t = @timed mapreduce(+, [2, 3]) do n_item
        policies = compute_policies(n_item, prm; UCB_PARAMS...)
        histograms = make_histograms(policies, prm.β_μ, LIKELIHOOD_PARAMS.n_sim_hist)
        trials = filter(x->iseven(x.trial), load_dataset(n_item))
        -likelihood(trials, histograms)[1]
    end
    println(round.(x; sigdigits=3), " => ", round(cl; digits=1), " ($(round(Int, t))s)")
    cl
end

results = pmap(best) do prm
    optimize(floss, collect(prm); iterations=30, store_trace=true, extended_trace=true)
end

# %% --------
using ProgressMeter
tuned = map(results) do res
    res.minimizer
end
initial = map(collect, best)
new_loss = @showprogress pmap(floss, hcat(initial, tuned))


# %% ==================== Prior only ====================
best = deserialize("$BASE_DIR/best_parameters/joint-fit")

results = pmap(best) do prm
    both_policies = map([2, 3]) do n_item 
        compute_policies(n_item, prm; UCB_PARAMS...)
    end
    optimize(0, 1; iterations=50) do β_μ
        cl, t = @timed mapreduce(+, [2, 3], both_policies) do n_item, policies
            histograms = make_histograms(policies, β_μ, LIKELIHOOD_PARAMS.n_sim_hist)
            trials = filter(x->iseven(x.trial), load_dataset(n_item))
            -likelihood(trials, histograms)[1]
        end
        println(round(β_μ; digits=3), " => ", round(cl; digits=1), " ($(round(Int, t))s)")
        cl
    end
end

# %% ==================== Check for benefit ====================

tuned = map(best, results) do prm, res
    (prm..., β_μ=res.minimizer)
end
using ProgressMeter
new_loss = @showprogress pmap(hcat(best, tuned)) do prm
    both_policies = map([2, 3]) do n_item 
         compute_policies(n_item, prm; UCB_PARAMS...)
    end
    mapreduce(+, [2, 3], both_policies) do n_item, policies
        histograms = make_histograms(policies, prm.β_μ, LIKELIHOOD_PARAMS.n_sim_hist)
        trials = filter(x->iseven(x.trial), load_dataset(n_item))
        -likelihood(trials, histograms)[1]
    end
end

