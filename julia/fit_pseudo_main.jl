include("results.jl")
include("pseudo_likelihood.jl")
include("box.jl")


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
    "dataset"
        required = true
        range_tester = x -> x in ("two", "three")
    "index"
        required = true
        range_tester = x -> x in ("even", "odd", "all")
end

args = parse_args(s)

dataset = args["dataset"]
const results = Results("$(dataset)_items_fixed")

println("Fitting dataset with $dataset items.")
all_trials = map(sort_value, load_dataset(dataset))

index = Dict(
    "odd" => 1:2:length(all_trials),
    "even" => 2:2:length(all_trials),
    "all" => 1:length(all_trials),
)[args["index"]]

fit_trials = all_trials[index]
N_ITEM = length(fit_trials[1].value)
μ_emp, σ_emp = juxt(mean, std)(flatten(fit_trials.value))

space = Box(
    :n_arm => N_ITEM,
    :sample_time => 100,
    :α => (50, 1000, :log),
    :σ_obs => (1, 10),
    :sample_cost => (1e-3, 5e-2, :log),
    :switch_cost => (1e-3, 1e-1, :log),
    :σ_rating => args["rating_noise"] ? (0., 1.) : 0.,
    :µ => args["fit_mu"] ? (0, μ_emp) : μ_emp,
    :σ => σ_emp,
)

# @assert all(issorted(t.value) for t in rank_trials)

if args["propfix"]
    metrics = [
        Metric(total_fix_time, 5, fit_trials),
        Metric(n_fix, Binning([0; 2:7; Inf])),
        Metric(t->t.choice, Binning(1:N_ITEM+1)),
        Metric(t->propfix(t)[1], 5, fit_trials),
        Metric(t->propfix(t)[end], 5, fit_trials)
    ]
else
    metrics = [
        Metric(total_fix_time, 10, fit_trials),
        Metric(n_fix, Binning([0; 2:7; Inf]), fit_trials),
        Metric(rank_chosen, Binning(1:N_ITEM+1)),
        # Metric(top_fix_proportion, 10)
    ]
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
    metrics = metrics,
)

bmps_kws = (
    n_iter=500,
)

@show opt_kws
@show like_kws
@show bmps_kws

let
    save(results, :opt_kws, opt_kws)
    save(results, :like_kws, like_kws)
    save(results, :metrics, metrics)
    save(results, :bmps_kws, bmps_kws)
    save(results, :index, index)
    save(results, :space, space)
end

loss_iter = 0

function loss(prm::Params)
    m = MetaMDP(prm)
    policy = optimize_bmps(m; α=prm.α, bmps_kws...)
    likelihood, ε, baseline = total_likelihood(policy, prm, fit_trials; like_kws...)
    save(results, Symbol(string("loss_", lpad(loss_iter, 3, "0"))),
         (prm=prm, policy=policy, ε=ε, likelihood=likelihood);
         verbose=false)

    max_loss = 2
    ll = isfinite(likelihood) ? min(likelihood / baseline, max_loss) : max_loss
    @printf "%.2f   %.4f\n" ε ll
    ll
end

function loss(x::Vector{Float64})
    global loss_iter += 1
    print("($loss_iter)  ", round.(x; digits=3), "  =>  ")
    loss(Params(space(x)))
end

function fit(opt)
    @info "Begin fitting" opt_kws like_kws
    for i in 1:4
        boptimize!(opt)
        find_model_max!(opt)
        prm = opt.model_optimizer |> space |> Params
        save(results, Symbol(string("mle_", loss_iter)), prm)
        save(results, :gp_model, opt.model)
        ℓ = -log.(opt.model.kernel.iℓ2) / 2 # log length scales
        loss = opt.model_optimum
        @info "Iteration $loss_iter" loss prm ℓ
    end
end

function reoptimize(prm::Params; N=16)
    policies = asyncmap(1:N) do i
        m = MetaMDP(prm)
        optimize_bmps(m; α=prm.α)
    end
    save(results, :reopt, policies)

    reopt_like = asyncmap(policies) do policy
        total_likelihood(policy, prm fit_trials; like_kws...)
    end
    save(results, :reopt_like, reopt_like)

    test_index = setdiff(eachindex(trials), like_kws.index)
    test_trials = all_trials[test_index]
    test_like = asyncmap(policies) do policy
        total_likelihood(policy, prm, test_trials; like_kws...)
    end
    save(results, :test_like, test_like)
end

opt = gp_minimize(loss, n_free(space); run=false, verbose=false, opt_kws...)

let
    fit(opt)
    find_model_max!(opt)
    mle = opt.model_optimizer |> space |> Params
    save(results, :mle, mle)
    save(results, :opt, opt)
    reoptimize(mle)
end