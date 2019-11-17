using Distributed

include("results.jl")
include("pseudo_likelihood.jl")

@assert @isdefined(res)
@assert @isdefined(args)

display(args); println()
# %% ====================  ====================
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
    like_kws = (
        fit_ε = !args["fix_eps"],
        max_ε = 0.5,
        n_sim_hist = args["n_sim_hist"]
    )
    bmps_kws = (
        n_iter=args["bmps_iter"],
    )
    opt_kws = (
        iterations=10_000,
        init_iters=100,
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

    clip_loss(loss, max_loss) = isfinite(loss) ? min(max_loss, loss) : max_loss
end

let
    save(res, :args, args)
    save(res, :opt_kws, opt_kws)
    save(res, :like_kws, like_kws)
    save(res, :datasets, datasets)
    save(res, :bmps_kws, bmps_kws)
    save(res, :fold, args["fold"])
    save(res, :space, space)
end


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


loss_iter = 0
function loss(prm::Params)
    losses = asyncmap(datasets) do d
        m = MetaMDP(d.n_item, prm)
        policy = optimize_bmps(m; α=prm.α, bmps_kws...)
        like, ε, baseline = likelihood(d, policy, prm)
        like / baseline
    end
    loss = clip_loss(sum(losses), 2 * length(losses))
    return loss
end

function loss(x::Vector{Float64}; verbose=true)
    global loss_iter += 1
    if verbose
        @time y = loss(Params(space(x)))
        println("($loss_iter)  ", round.(x; digits=3), "  =>  ", round(y; digits=2))
        flush(stdout)
        return y
    else
        return loss(Params(space(x)))
    end
end

function preopt_init(datasets, res_name)
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

using Sobol
function sobol_init()
    println("Initializing with Sobol sequence")
    seq = SobolSeq(5)
    xs = [next!(seq) for _ in 1:opt_kws.init_iters]
    ys = asyncmap(xs) do x
        loss(x; verbose=false)
    end
    combinedims(xs), ys
end

function record_mle(opt, i)
    find_model_max!(opt)
    prm = opt.model_optimizer |> space |> Params
    save(res, :mle, prm)
    save(res, Symbol("mle_$i"), prm)
    save(res, :xy, (x=opt.model.x, y=opt.model.y))
    ℓ = -log.(opt.model.kernel.iℓ2) / 2 # log length scales
    loss = opt.model_optimum
    @info "Iteration $loss_iter" loss prm repr(ℓ)
    flush(stdout)
end

function fit(opt)
    @info "Begin fitting" opt_kws like_kws
    maxiterations!(opt, args["save_freq"])  # set maxiterations for the next call
    n_loop = Int(args["fit_iter"] / args["save_freq"])
    for i in 1:n_loop
        boptimize!(opt)
        record_mle(opt, i)
        # save(res, :gp_model, opt.model)
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


