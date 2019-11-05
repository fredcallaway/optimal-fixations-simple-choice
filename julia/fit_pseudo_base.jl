using Distributed

include("results.jl")
include("pseudo_likelihood.jl")
include("box.jl")

@assert @isdefined(res)
@assert @isdefined(args)

function train_test_split(trials, fold)
    train_idx = Dict(
        "odd" => 1:2:length(trials),
        "even" => 2:2:length(trials),
        "all" => 1:length(trials),
    )[fold]
    test_idx = setdiff(eachindex(trials), train_idx)
    (train=trials[train_idx], test=trials[test_idx])
end

both_trials = map(["two", "three"]) do n_item
    train_test_split(map(sort_value, load_dataset(n_item)), args["fold"])
end

empirical_prior(trials) = juxt(mean, std)(flatten(trials.value))
emp_priors = map(both_trials) do trials
    empirical_prior(trials.train)
end

space = Box(
    :sample_time => 100,
    :α => (50, 300, :log),
    :σ_obs => (1, 6),
    :sample_cost => (.001, .01, :log),
    :switch_cost => (.01, .05, :log),
    :σ_rating => args["rating_noise"] ? (0., 1.) : 0.,
    # :µ => args["fitmu"] ? (0, μ_emp) : μ_emp,
    :β_μ => (0., 1.),
    :β_σ => 1.,  # FIXME
)

both_metrics = map(both_trials) do trials
    n_item = length(trials.train[1].value)
    if args["propfix"]
        metrics = [
            Metric(total_fix_time, 5, trials.train),
            Metric(n_fix, Binning([0; 2:7; Inf])),
            Metric(t->t.choice, Binning(1:n_item+1)),
        ]
        for i in 1:(n_item-1)
            push!(metrics, Metric(t->propfix(t)[i], 5, trials.train))
        end
        return metrics
    else
        [
            Metric(total_fix_time, 10, trials.train),
            Metric(n_fix, Binning([0; 2:7; Inf])),
            Metric(rank_chosen, Binning(1:n_item+1)),
            # Metric(top_fix_proportion, 10)
        ]
    end
end

opt_kws = (
    iterations=100,
    init_iters=100,
    acquisition="ei",
    optimize_every=5,
    acquisition_restarts=200,
    noisebounds=[-4, 1],
)

like_kws = (
    fit_ε = !args["fix_eps"],
    max_ε = 0.5,
    n_sim_hist = args["n_sim_hist"]
)

bmps_kws = (
    n_iter=args["bmps_iter"],
)

let
    save(res, :args, args)
    save(res, :opt_kws, opt_kws)
    save(res, :like_kws, like_kws)
    save(res, :metrics, both_metrics)
    save(res, :bmps_kws, bmps_kws)
    save(res, :fold, args["fold"])
    save(res, :space, space)
end

loss_iter = 0
function loss(prm::Params)
    losses = asyncmap(1:2) do i
        m = MetaMDP(i+1, prm)
        policy = optimize_bmps(m; α=prm.α, bmps_kws...)
        μ_emp, σ_emp = emp_priors[i]
        lk_prm = (
            μ=prm.β_μ * μ_emp,
            σ=prm.β_σ * σ_emp,
            σ_rating=prm.σ_rating,
        )
        likelihood, ε, baseline = total_likelihood(policy, lk_prm, both_trials[i].train;
            metrics=both_metrics[i], like_kws...)
        likelihood / baseline
    end
    max_loss = 4.
    loss = min(max_loss, sum(losses))
    if !isfinite(loss)
        loss = max_loss
    end
    @printf "%.3f + %.3f = %.3f\n" losses[1] losses[2] loss
    loss
end

function loss(x::Vector{Float64})
    # loss_iter = @isdefined(opt) ? length(opt.model.y) + 1 : 0
    global loss_iter += 1

    print("($loss_iter)  ", round.(x; digits=3), "  =>  ")
    @time loss(Params(space(x)))
end

function fit(opt; n_iter=4)
    @info "Begin fitting" opt_kws like_kws
    for i in 1:n_iter
        boptimize!(opt)
        find_model_max!(opt)
        prm = opt.model_optimizer |> space |> Params
        save(res, :mle, prm)
        save(res, :gp_model, opt.model)
        ℓ = -log.(opt.model.kernel.iℓ2) / 2 # log length scales
        loss = opt.model_optimum
        @info "Iteration $loss_iter" loss prm repr(ℓ)
    end
end

function reoptimize(prm::Params; N=16)
    reopt = map(1:2) do i
        policies = asyncmap(1:N) do j
            m = MetaMDP(i+1, prm)
            optimize_bmps(m; α=prm.α)
        end

        μ_emp, σ_emp = emp_priors[i]
        lk_prm = (
            μ=prm.β_μ * μ_emp,
            σ=prm.β_σ * σ_emp,
            σ_rating=prm.σ_rating,
        )

        train_like = asyncmap(policies) do policy
            total_likelihood(policy, like_prm, both_trials[i].train;
                metrics=both_metrics[i], like_kws...)
        end

        test_like = asyncmap(policies) do policy
            total_likelihood(policy, like_prm, both_trials[i].test;
                metrics=both_metrics[i], like_kws...)
        end
        (policies=policies, train_like=train_like, test_like=test_like)
    end

    save(res, :reopt, reopt)
end