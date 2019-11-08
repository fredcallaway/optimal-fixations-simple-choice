using Distributed

include("results.jl")
include("pseudo_likelihood.jl")
include("box.jl")

@assert @isdefined(res)
@assert @isdefined(args)

display(args); println()
# %% ====================  ====================
function build_metrics(trials)
    n_item = length(trials[1].value)
    if args["propfix"]
        metrics = [
            Metric(total_fix_time, 5, trials),
            Metric(n_fix, Binning([0; 2:7; Inf])),
            Metric(t->t.choice, Binning(1:n_item+1)),
        ]
        for i in 1:(n_item-1)
            push!(metrics, Metric(t->propfix(t)[i], 5, trials))
        end
        return metrics
    else
        [
            Metric(total_fix_time, 10, trials),
            Metric(n_fix, Binning([0; 2:7; Inf])),
            Metric(rank_chosen, Binning(1:n_item+1)),
            # Metric(top_fix_proportion, 10)
        ]
    end
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

datasets = [build_dataset(args["dataset"], args["subject"])]
# datasets = [build_dataset("two", -1), build_dataset("three", -1)]

println("Dataset sizes: ", join(map(d->length(d.train_trials), datasets), " "))
space = Box(
    :sample_time => 100,
    :α => (50, 300, :log),
    :σ_obs => (1, 10),
    :sample_cost => (.001, .01, :log),
    :switch_cost => (.01, .1, :log),
    :σ_rating => args["rating_noise"] ? (0., 1.) : 0.,
    # :µ => args["fitmu"] ? (0, μ_emp) : μ_emp,
    :β_μ => (0., 1.),
    :β_σ => 1.,  # FIXME
)

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
    save(res, :datasets, datasets)
    save(res, :bmps_kws, bmps_kws)
    save(res, :fold, args["fold"])
    save(res, :space, space)
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

loss_iter = 0
function loss(prm::Params)
    losses = asyncmap(datasets) do d
        m = MetaMDP(d.n_item, prm)
        policy = optimize_bmps(m; α=prm.α, bmps_kws...)

        like, ε, baseline = likelihood(d, policy, prm)
        like / baseline
    end
    max_loss = 4.
    loss = min(max_loss, sum(losses))
    if !isfinite(loss)
        loss = max_loss
    end
    length(losses) > 1 && print(round.(losses; digits=2), "  ")
    println(round(loss; digits=2))
    # @printf "%.3f + %.3f = %.3f\n" losses[1] losses[2] loss
    loss
end

function loss(x::Vector{Float64})
    # loss_iter = @isdefined(opt) ? length(opt.model.y) + 1 : 0
    global loss_iter += 1

    print("($loss_iter)  ", round.(x; digits=3), "  =>  ")
    @time y = loss(Params(space(x)))
    flush(stdout)
    y
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
        flush(stdout)
    end
end

function reoptimize(prm::Params; N=16)
    reopt = map(datasets) do d
        policies = asyncmap(1:N) do j
            m = MetaMDP(d.n_item, prm)
            optimize_bmps(m; α=prm.α)
        end

        train_like = asyncmap(policies) do policy
            likelihood(d, policy, prm)[1:3]
        end

        test_like = asyncmap(policies) do policy
            likelihood(d, policy, prm; test=true)[1:3]
        end
        (policies=policies, train_like=train_like, test_like=test_like)
    end

    save(res, :reopt, reopt)
end


