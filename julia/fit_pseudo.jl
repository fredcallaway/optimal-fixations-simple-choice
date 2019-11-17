include("results.jl")
using ArgParse

using Logging; global_logger(SimpleLogger(stdout, Logging.Info))

s = ArgParseSettings()
@add_arg_table s begin
    # "--propfix"
    #     action = :store_true
    "--fix_eps"
        action = :store_true
    "--rating_noise"
        action = :store_true
    # "--fitmu"
    #     action = :store_true
    "--bmps_iter"
        arg_type = Int
        default = 500
    "--fit_iter"
        arg_type = Int
        default = 1000
    "--save_freq"
        arg_type = Int
        default = 10
    "--n_sim_hist"
        arg_type = Int
        default = 10000
    "--dataset"
        arg_type = String
        default = "both"
        range_tester = x -> x in ("two", "three", "both")
    "--subject"
        arg_type = Int
        default = -1  # fit all subjects
    "--fold"
        arg_type = String
        default = "even"
        # range_tester = x -> x in ("even", "odd", "all")
    "--preopt"
        arg_type = Int
        default = 0
    "--res"
        arg_type = String
        default = "test"
    "--init"
        arg_type = String
        default = ""
    "--hist_bins"
        arg_type = Int
        default=5
end

args = parse_args(s)
args["propfix"] = true  # old arguments, now with fixed values
args["fitmu"] = true
res = Results(args["res"])

println("Initializing...")
@time include("fit_pseudo_base.jl")
flush(stdout)
preopt = args["preopt"]
init = args["init"]
if preopt == 0
    init_Xy = (init == "") ? sobol_init() : preopt_init(datasets, init)

    opt = gp_minimize(loss, n_free(space);
        run=false, verbose=false,
        opt_kws...,
        init_Xy=init_Xy
    )
    record_mle(opt, 0)

    @time fit(opt)
    find_model_max!(opt)
    mle = opt.model_optimizer |> space |> Params
    save(res, :mle, mle)
    # save(res, :opt, opt)
    println("Computing policies for MLE")
    @time reoptimize(mle)
else
    println("Preoptimizing #$(preopt)")

    seq = SobolSeq(n_free(space))
    skip(seq, preopt-1; exact=true)
    x = next!(seq)
    prm = Params(space(x))
    @assert prm.σ_rating == 0

    @time policies = map(datasets) do d
        m = MetaMDP(d.n_item, prm)
        policy = optimize_bmps(m; α=prm.α, bmps_kws...)
    end

    save(res, :sobol_i, preopt)
    save(res, :policies, policies)

    # vs = unique(sort(t.value) for t in [d.train_trials; d.test_trials]);
    # sort!(vs, by=v->abs(v[1] - v[2]))  # slowest trials first for parallel efficiency
    # sims = pmap(vs) do v
    #     v => [sim_one(policy, prm, v) for _ in 1:10_0]
    # end |> Dict

    # save(res, :loss, loss(prm))
    # println("Computing loss for sobol $(preopt)")
    # seq = SobolSeq(n_free(space))
    # skip(seq, preopt-1; exact=true)
    # x = next!(seq)
    # prm = Params(space(x))
    # println(prm)
    # save(res, :sobol, preopt)
    # save(res, :prm, prm)
    # save(res, :loss, loss(prm))
end
