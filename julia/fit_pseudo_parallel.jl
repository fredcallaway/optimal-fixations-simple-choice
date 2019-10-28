using Distributed

@time begin
    include("results.jl")
    include("pseudo_likelihood.jl")
    include("box.jl")
end

# push!(ARGS, "both", "odd")
# push!(ARGS, "--propfix", "--fit_mu")
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--propfix"
        action = :store_true
    "--fix_eps"
        action = :store_true
    "--rating_noise"
        action = :store_true
    "--fit_mu"
        action = :store_true
    "--bmps_iter"
        arg_type = Int
        default = 500
    "--n_sim_hist"
        arg_type = Int
        default = 10000
    "dataset"
        required = true
        range_tester = x -> x in ("two", "three", "both")
    "fold"
        required = true
        range_tester = x -> x in ("even", "odd", "all")
    "job"
        required = true
        arg_type = Int
end

args = parse_args(s)
dataset = args["dataset"]
println("Fitting dataset with $dataset items.")
results = Results("$(dataset)_items_fixed_parallel")

function train_test_split(trials, fold)
    train_idx = Dict(
        "odd" => 1:2:length(trials),
        "even" => 2:2:length(trials),
        "all" => 1:length(trials),
    )[fold]
    test_idx = setdiff(eachindex(trials), train_idx)
    (train=trials[train_idx], test=trials[test_idx])
end

all_trials = map(["two", "three"]) do n_item
    train_test_split(map(sort_value, load_dataset(n_item)), args["fold"])
end

# empirical_prior(trials) = juxt(mean, std)(flatten(trials.value))
# empirical_prior(all_trials[1].train)
# empirical_prior(all_trials[2].train)

@assert args["fit_mu"]
# μ_emp, σ_emp = juxt(mean, std)(flatten(fit_trials.value))


space = Box(
    :sample_time => 100,
    :α => (50, 300, :log),
    :σ_obs => (1, 6),
    :sample_cost => (.001, .01, :log),
    :switch_cost => (.01, .05, :log),
    :σ_rating => args["rating_noise"] ? (0., 1.) : 0.,
    # :µ => args["fit_mu"] ? (0, μ_emp) : μ_emp,
    :μ => (0,5),
    :σ => 2.55,  # FIXME
)

both_metrics = map(all_trials) do trials
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

@show opt_kws
@show like_kws
@show bmps_kws

let
    save(results, :opt_kws, opt_kws)
    save(results, :like_kws, like_kws)
    save(results, :metrics, both_metrics)
    save(results, :bmps_kws, bmps_kws)
    save(results, :fold, args["fold"])
    save(results, :space, space)
end


function loss(prm::Params)
    losses = asyncmap(1:2) do i
        m = MetaMDP(i+1, prm)
        policy = optimize_bmps(m; α=prm.α, bmps_kws...)
        likelihood, ε, baseline = total_likelihood(policy, prm, all_trials[i].train;
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


seq = SobolSeq(n_free(space))
skip(seq, args["job"]-1; exact=true)
x = next!(seq)
prm = Params(space(x))
println(prm)
save(results, :prm, prm)

@time save(results, :loss, loss(prm))

