include("elastic.jl")
if get(ARGS, 1, "") == "master"
    addprocs(topology=:master_worker)
end

@everywhere begin
    include("model_base.jl")
    include("box.jl")
    include("optimize_bmps.jl")

    const space = Box(
        :obs_sigma => (1, 10),
        :sample_cost => (1e-4, 1e-2, :log),
        :switch_cost => (1, 60),
    )

    # const NAME = "test"
    # const N_RAND = 100
    # const N_ITER = 8
    # const N_ROLL = 10
    const NAME = "rand_bmps"
    const N_RAND = 1000
    const N_ITER = 800
    const N_ROLL = 1000
end



if get(ARGS, 1, "") == "worker"
    start_worker()
elseif get(ARGS, 1, "") == "master"
    start_master(wait=0)

    println("Press enter to continue")
    ready = readline()

    using Sobol
    sobol_points = let
        seq = SobolSeq(n_free(space))
        skip(seq, N_RAND)
        Iterators.take(seq, N_RAND)
    end

    smap(sobol_points) do x
        results = Results(NAME)
        mdp = MetaMDP(;space(x)...)
        save(results, :mdp, mdp; verbose=false)
        @time policy, opt = optimize(mdp; n_iter=N_ITER, n_roll=N_ROLL, parallel=false)
        save(results, :policy, policy; verbose=false)
        save(results, :opt, opt; verbose=false)

        observed = round.(opt.observed_optimizer; digits=3) => round(opt.observed_optimum; digits=5)
        model = round.(opt.model_optimizer; digits=3) => round(opt.model_optimum; digits=5)
        println(string(
            "────────────────────────────────────",
            "\nobserved: ", observed,
            "\nmodel:    ", model
        ))
        # @info "────────────────────────────────────" observed model
    end
end


