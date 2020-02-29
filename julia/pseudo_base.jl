function build_metrics(trials)
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

function build_dataset(num, subject)
    trials = map(sort_value, load_dataset(num, subject))
    train, test = train_test_split(trials, LIKELIHOOD_PARAMS.test_fold)
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
