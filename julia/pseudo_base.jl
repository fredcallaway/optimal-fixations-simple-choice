function build_metrics(trials)
    n_item = length(trials[1].value)
    hb = args["hist_bins"]
    metrics = [
        Metric(total_fix_time, hb, trials),
        Metric(n_fix, Binning([0; 2:hb; Inf])),
        Metric(t->t.choice, Binning(1:n_item+1)),
    ]
    if args["propfix"]
        for i in 1:(n_item-1)
            push!(metrics, Metric(t->propfix(t)[i], hb, trials))
        end
    end
    return metrics
    # else
    #     [
    #         Metric(total_fix_time, hb, trials),
    #         Metric(n_fix, Binning([0; 2:7; Inf])),
    #         Metric(rank_chosen, Binning(1:n_item+1)),
    #         # Metric(top_fix_proportion, 10)
    #     ]
    # end
end

function build_dataset(num, subject)
    trials = map(sort_value, load_dataset(num, subject))
    train, test = train_test_split(trials, args["fold"])
    μ_emp, σ_emp = empirical_prior(trials)
    (
        subject=subject,
        n_item = length(trials[1].value),
        train_trials = train,
        test_trials = test,
        μ_emp = μ_emp,
        σ_emp = σ_emp,
        metrics = build_metrics(trials)
    )
end

like_params(d, prm) = (
    μ=prm.β_μ * d.μ_emp,
    σ=prm.β_σ * d.σ_emp,
    σ_rating=prm.σ_rating,
)

function likelihood(d, policy, prm; test=false, kws...)
    trials = test ? d.test_trials : d.train_trials
    total_likelihood(
        policy, like_params(d, prm), trials;
        metrics=d.metrics, like_kws..., kws...
    )
end
