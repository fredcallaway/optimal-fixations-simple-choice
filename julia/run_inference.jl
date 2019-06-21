
include("elastic.jl")

if get(ARGS, 1, "") == "master"
    include("results.jl")
    include("box.jl")
    include("gp_min.jl")
    addprocs(topology=:master_worker)
    # addprocs([("griffiths-gpu01.pni.princeton.edu", :auto)], tunnel=true, topology=:master_worker)
    println(nprocs(), " processes")
    results = Results("inference")
end


@everywhere begin
    include("inference.jl")

    config = (
        sample_time = 100,
        n_particle = 1000,
        iterations = 1000,
        reweight = false,
        acquisition = "ei",
    )
    const data = let
        # put longest trials first to make parallelization more efficient
        dd = Datum.(trials, config.sample_time)
        # println("SUBSETTING DATA!!")
        # dd = dd[1:96]
        order = sortperm(map(d->-length(d.samples), dd))
        dd[order]
    end
end
# %% ====================  ====================


if get(ARGS, 1, "") == "worker"
    start_worker()
elseif get(ARGS, 1, "") == "master"
    start_master(wait=false)
    save(results, :config, config)
    println(config)

    space = Box(
        :α => (10, 100, :log),
        :obs_sigma => (1, 60),
        # :sample_cost => (0.0004, 0.002, :log),
        :sample_cost => (1e-4, 1e-2, :log),
        :switch_cost => (1, 60),
        :µ => μ_emp,
        :σ => σ_emp,
        # (0, 2 * μ_emp),
        # :σ => (σ_emp / 4, 4 * σ_emp),
    )
    save(results, :space, space)
    println(space)

    const RAND_LOGP = sum(rand_logp.(data))
    const MAX_LOSS = 10

    function plogp(prm, n_particle=config.n_particle)
        smap(eachindex(data)) do i
            logp(prm, data[i], config.reweight, n_particle)
        end |> sum
    end

    function loss(x, n_particle=config.n_particle)
        prm = Params(;space(x)...)
        100 * min(MAX_LOSS, plogp(prm, n_particle) / RAND_LOGP)
    end

    println("Begin GP minimize")
    opt = gp_minimize(loss, n_free(space);
                      iterations=config.iterations,
                      acquisition=config.acquisition,
                      file=path(results, :opt))

    println("observed: ", round.(opt.observed_optimizer; digits=3),
            " => ", round(opt.observed_optimum; digits=5))
    println("model:    ", round.(opt.model_optimizer; digits=3),
            " => ", round(opt.model_optimum; digits=5))

    f_mod = @show loss(opt.model_optimizer, 10000)
    f_obs = @show loss(opt.observed_optimizer, 10000)
    best = f_obs < f_mod ? opt.observed_optimizer : opt.model_optimizer

    prm = Params(;space(best)...)
    println("MLE ", prm)

    lp = plogp(prm, 10000)
    println("Log Likelihood: ", lp)
    # println("BIC: ", log(N_OBS) * n_free(space) - 2 * lp)
    # println("AIC: ", 2 * n_free(space) - 2 * lp)

    save(results, :mle, (
        policy=SoftBlinkered(prm),
        prior=(prm.µ, prm.σ),
        logp=lp
    ))


    # %% ==================== Examine loss function around minimum ====================
    using JSON

    println("Explore loss function near discovered minimum.")
    diffs = -0.1:0.02:0.1

    cross = map(1:n_free(space)) do i
        asyncmap(diffs) do d
            x = copy(best)
            x[i] += d
            try
                loss(x)
            catch
                NaN
            end
        end
    end

    open("$(dir(results))/cross.json", "w+") do f
        write(f, json(cross))
    end

    cross = map(1:n_free(space)) do i
        asyncmap(0:0.1:1) do d
            x = copy(best)
            x[i] = d
            try
                loss(x)
            catch
                NaN
            end
        end
    end

    open("$(dir(results))/cross_full.json", "w+") do f
        write(f, json(cross))
    end

end

