include("results.jl")
using ArgParse

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
        default = "fit_pseudo"
end


args = parse_args(s)
# %% ==================== Debugging ====================
# println("WARNING OVERRIDING ARGUMENTS")
args["propfix"] = true
args["fitmu"] = true
# %% ====================  ====================
res = Results(args["res"])

println("Initializing...")
@time include("fit_pseudo_base.jl")
flush(stdout)
preopt = args["preopt"]
if preopt == 0
    println("Running BayesOpt fitting")
    opt = gp_minimize(loss, n_free(space); run=false, verbose=false, opt_kws...)
    @time fit(opt)
    find_model_max!(opt)
    mle = opt.model_optimizer |> space |> Params
    save(res, :mle, mle)
    save(res, :opt, opt)
    println("Computing policies for MLE")
    @time reoptimize(mle)
else
    println("Preoptimizing #$(preopt)")

    seq = SobolSeq(n_free(space))
    skip(seq, preopt-1; exact=true)
    x = next!(seq)
    prm = Params(space(x))

    policies = asyncmap(datasets) do d
        m = MetaMDP(d.n_item, prm)
        policy = optimize_bmps(m; α=prm.α, bmps_kws...)
        # @assert prm.σ_rating == 0
        # sim = sim_one(policy, prm, v)
    end

    save(res, :sobol_i, preopt)
    save(res, :policy, policy)

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
