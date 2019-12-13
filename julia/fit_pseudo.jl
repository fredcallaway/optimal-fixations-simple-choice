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
    "--fit_mu"
        action = :store_true
    "--bmps_iter"
        arg_type = Int
        default = 500
    "--bmps_roll"
        arg_type = Int
        default = 10000
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
    "--sobol"
        arg_type = Int
        default = 0
    "--res"
        arg_type = String
        default = "test"
    "--init"
        arg_type = String
        default = ""
    "--n_init"
        arg_type = Int
        default = 100
    "--hist_bins"
        arg_type = Int
        default=5
    "--n_inner"
        arg_type = Int
        default = 2
end



args = parse_args(s)

args["propfix"] = true  # old arguments, now with fixed values
# args["fitmu"] = true
res = Results(args["res"])

println("Initializing...")
@time include("fit_pseudo_base.jl")
save_args()
flush(stdout)
sobol = args["sobol"]
init = args["init"]

if sobol == 0
    # init_Xy = (init == "") ? sobol_init() : preopt_init(datasets, init)
    init_Xy = (init == "") ? sobol_init() : precomputed_init(init)

    opt = gp_minimize(loss, n_free(outer_space);
        run=false, verbose=false,
        opt_kws...,
        init_Xy=init_Xy
    )
    record_mle(opt, 0)

    @time mle = fit(opt)
    # save(res, :opt, opt)
    println("Computing policies for MLE")
    @time reoptimize(mle)
else
    println("Precomputing loss for sobol #$(sobol)")

    seq = SobolSeq(n_free(outer_space))
    skip(seq, sobol-1; exact=true)
    x = next!(seq)

    prm_outer = namedtuple(outer_space(x))
    policies, pol_time = @timed optimal_policies(prm_outer)
    (x_inner, fx), inner_time = @timed inner_optimize(policies)
    prm_inner = x_inner |> inner_space |> namedtuple

    println(
        round.([x; x_inner]; digits=3),
        @sprintf("  =>  %.2f  (%ds + %ds)", fx, pol_time, inner_time)
    )
    flush(stdout)

    save(res, :sobol_i, sobol)
    save(res, :prm_outer, prm_outer)
    save(res, :prm_inner, prm_inner)
    save(res, :policies, policies)
    save(res, :loss, fx)

    # vs = unique(sort(t.value) for t in [d.train_trials; d.test_trials]);
    # sort!(vs, by=v->abs(v[1] - v[2]))  # slowest trials first for parallel efficiency
    # sims = pmap(vs) do v
    #     v => [sim_one(policy, prm, v) for _ in 1:10_0]
    # end |> Dict

    # save(res, :loss, loss(prm))
    # println("Computing loss for sobol $(sobol)")
    # seq = SobolSeq(n_free(space))
    # skip(seq, sobol-1; exact=true)
    # x = next!(seq)
    # prm = Params(space(x))
    # println(prm)
    # save(res, :sobol, sobol)
    # save(res, :prm, prm)
    # save(res, :loss, loss(prm))
end
