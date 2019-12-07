using Distributed

include("results.jl")
include("pseudo_likelihood.jl")

# @assert @isdefined(res)  # only necessary for functions that save
@assert @isdefined(args)

display(args); println()
# %% ==================== Setup ====================
function build_metrics(trials)
    n_item = length(trials[1].value)
    if args["propfix"]
        hb = args["hist_bins"]
        metrics = [
            Metric(total_fix_time, hb, trials),
            Metric(n_fix, Binning([0; 2:7; Inf])),
            Metric(t->t.choice, Binning(1:n_item+1)),
        ]
        for i in 1:(n_item-1)
            push!(metrics, Metric(t->propfix(t)[i], hb, trials))
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


datasets = if args["dataset"] == "both"
    @assert args["subject"] == -1
    [build_dataset("two", -1), build_dataset("three", -1)]
else
    [build_dataset(args["dataset"], args["subject"])]
end

println("Dataset sizes: ", join(map(d->length(d.train_trials), datasets), " "))

@everywhere begin
    args = $args
    include("box.jl")

    outer_space = Box(
        :sample_time => 100,
        :σ_obs => (1, 10),
        :sample_cost => (.001, .01, :log),
        :switch_cost => (.01, .1, :log),
    )
    inner_space = Box(
        :α=>(50., 500., :log),
        :β_μ=>(0.,1.),
        :β_σ=>1.,
        :σ_rating => args["rating_noise"] ? (0., 1.) : 0.,
    )
    like_kws = (
        fit_ε = !args["fix_eps"],
        max_ε = 0.5,
        n_sim_hist = args["n_sim_hist"]
    )
    bmps_kws = (
        n_iter=args["bmps_iter"],
        n_roll=args["bmps_roll"],
        α=500.
    )
    opt_kws = (
        iterations=10_000,
        init_iters=args["n_init"],
        acquisition="ei",
        optimize_every=5,
        acquisition_restarts=200,
        noisebounds=[-4, 1],
    )

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

    clip_loss(loss, max_loss=2) = isfinite(loss) ? min(max_loss, loss) : max_loss
end

function save_args()
    save(res, :args, args, verbose=false)
    save(res, :opt_kws, opt_kws, verbose=false)
    save(res, :like_kws, like_kws, verbose=false)
    save(res, :datasets, datasets, verbose=false)
    save(res, :bmps_kws, bmps_kws, verbose=false)
    save(res, :fold, args["fold"], verbose=false)
    save(res, :outer_space, outer_space, verbose=false)
    save(res, :inner_space, inner_space, verbose=false)
end

# %% ==================== Loss ====================

change_α(policy, α) = BMPSPolicy(policy.m, policy.θ, float(α))

function inner_loss(policies, x)
    lp = namedtuple(inner_space(collect(x)))
    mapreduce(+, datasets, policies) do d, policy
        pol = change_α(policy, lp.α)
        logp, ε, baseline = likelihood(d, pol, lp);
        clip_loss(logp / baseline)
    end
end

function inner_optimize(policies)
    # randomly initialzie
    x1 = rand(); x2 = rand()
    for i in 1:args["n_inner"]
        # optimize the decision temperature given that prior
        x1 = optimize(x1->inner_loss([x1, x2]), 0, 1, abs_tol=0.01, iterations=10).minimizer
        # optimize the prior given that temperature
        x2 = optimize(x2->inner_loss([x1, x2]), 0, 1, abs_tol=0.01, iterations=10).minimizer
    end
    # we don't seem to improve by optimizing more than that...
    best = [x1, x2]
    best, inner_loss(best)
end



function optimal_policies(prm)
    asyncmap(datasets) do d
        m = MetaMDP(d.n_item, prm.σ_obs, prm.sample_cost, prm.switch_cost)
        optimize_bmps(m; bmps_kws...)
    end
end

loss_iter = 0
function loss(x::Vector{Float64}; verbose=true)
    prm = namedtuple(outer_space(x))
    policies, pol_time = @timed optimal_policies(prm)
    (x_inner, fx), inner_time = @timed inner_optimize(policies)

    global loss_iter += 1
    if verbose
        println(
            "($loss_iter)  ", round.([x; x_inner]; digits=2),
            @sprintf("  =>  %.3f  (%ds + %ds)", fx, pol_time, inner_time)
        )
        flush(stdout)
    end
    fx
end


# %% ==================== Initialization ====================

function get_preopt_pols(res_name)
    seen = Set{Int}()
    asyncmap(get_results(res_name)) do res
        (exists(res, :sobol_i) && exists(res, :policies)) || return missing
        arg_ = load(res, :args)
        (arg_["dataset"] == args["dataset"]) || return missing

        si = load(res, :sobol_i)
        (si in seen) && return missing
        push!(seen, si)

        load(res, :policies)[1]
    end |> skipmissing |> collect
end



function preopt_init(datasets, res_name)
    error("BROKEN")
    println("Initializing with preoptimized policies")
    pols = get_preopt_pols(res_name)
    @time init = pmap(pols) do pol
        map((0:0.25:.75) .+ 0.25rand()) do β_μ
            prm = Params(
                α = pol.α,
                σ_obs = pol.m.σ_obs,
                sample_cost = pol.m.sample_cost,
                switch_cost = pol.m.switch_cost,
                β_µ = β_µ,
                β_σ = 1,
                sample_time = 100,
                σ_rating = 0.,
            )
            losses = map(datasets) do d
                lk, ε, base = likelihood(d, pol, prm, n_sim_hist=1000, parallel=false)
                lk / base
            end
            space(type2dict(prm)), clip_loss(sum(losses), 2 * length(losses))
        end
    end
    xs, y = invert(flatten(init))
    X = combinedims(xs)
    X, y
end

function precomputed_init(res_name)
    println("Initializing with precomputed losses.")
    xs, y = map(get_results(res_name)) do res
        exists(res, :loss) || return missing
        exists(res, :sobol_i) || return missing
        args2 = load(res, :args)
        args["dataset"] == args2["dataset"] || return missing
        args["fold"] == args2["fold"] || return missing
        # load(res, :prm_inner)
        prm = load(res, :prm_outer)
        x = outer_space(type2dict(prm))
        x, load(res, :loss)
    end |> skipmissing |> collect |> invert
    println("   found $(length(y)).")
    combinedims(xs), y
end

using Sobol
function sobol_init()
    println("Initializing with Sobol sequence")
    seq = SobolSeq(n_free(outer_space))
    xs = [next!(seq) for _ in 1:opt_kws.init_iters]
    ys = map(xs) do x
        loss(x; verbose=true)
    end
    combinedims(xs), ys
end


# %% ==================== Main functions ====================

function record_mle(opt, i)
    find_model_max!(opt)
    prm_outer = opt.model_optimizer |> outer_space |> namedtuple
    policies = optimal_policies(prm_outer)
    x_inner, fx = inner_optimize(policies)
    prm_inner = x_inner |> inner_space |> namedtuple

    prm = (n_obs=length(opt.model.y), prm_outer..., prm_inner...)
    save(res, :mle, prm)
    save(res, Symbol("mle_$i"), prm, verbose=false)
    save(res, :xy, (x=opt.model.x, y=opt.model.y), verbose=false)

    println("*"^60)
    ℓ = -log.(opt.model.kernel.iℓ2) / 2 # log length scales
    println("Iteration $loss_iter")
    println("  ", round.(-opt.model_optimizer; digits=3))
    println("  ", round(-opt.model_optimum; digits=3),
            "  ", round(fx; digits=3))
    println("  ", round.(ℓ; digits=1))
    print("  ")
    display(pairs(prm)); println()
    println("*"^60)
    flush(stdout)
    return prm
end

function fit(opt)
    @info "Begin fitting" opt_kws like_kws
    maxiterations!(opt, args["save_freq"])  # set maxiterations for the next call
    n_loop = Int(args["fit_iter"] / args["save_freq"])
    mle = nothing
    for i in 1:n_loop
        boptimize!(opt)
        mle = record_mle(opt, i)
        # save(res, :gp_model, opt.model)
    end
    return mle
end

function reoptimize(prm; N=16)
    reopt = map(datasets) do d
        policies = asyncmap(1:N) do j
            m = MetaMDP(d.n_item, prm.σ_obs, prm.sample_cost, prm.switch_cost)
            policy = optimize_bmps(m; bmps_kws...)
            change_α(policy, prm.α)
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


