include("results.jl")
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--propfix"
        action = :store_true
    "--fix_eps"
        action = :store_true
    "--rating_noise"
        action = :store_true
    "--fitmu"
        action = :store_true
    "--bmps_iter"
        arg_type = Int
        default = 500
    "--n_sim_hist"
        arg_type = Int
        default = 10000
    "--dataset"
        arg_type = String
        default = "both"
        range_tester = x -> x in ("two", "three", "both")
    "--fold"
        arg_type = String
        default = "even"
        range_tester = x -> x in ("even", "odd", "all")
    "--subject"
        arg_type = Int
        default = -1  # fit all subjects
    "--job"
        arg_type = Int
        default = 0
    "--res"
        arg_type = String
        default = "fit_pseudo"
end

args = parse_args(s)
res = Results(args["res"])

println("Initializing...")
@time include("fit_pseudo_base.jl")

job = args["job"]
if job == 0
    println("Running BayesOpt fitting.")
    opt = gp_minimize(loss, n_free(space); run=false, verbose=false, opt_kws...)
    fit(opt)
    find_model_max!(opt)
    mle = opt.model_optimizer |> space |> Params
    save(res, :mle, mle)
    save(res, :opt, opt)
    reoptimize(mle)
else
    println("Computing loss for sobol $(job)")
    seq = SobolSeq(n_free(space))
    skip(seq, job-1; exact=true)
    x = next!(seq)
    prm = Params(space(x))
    println(prm)
    save(res, :sobol, job)
    save(res, :prm, prm)
    save(res, :loss, loss(prm))
end
