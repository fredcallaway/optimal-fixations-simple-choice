
include("elastic.jl")

if get(ARGS, 1, "") == "master"
    include("results.jl")
    addprocs(topology=:master_worker)
    # addprocs([("griffiths-gpu01.pni.princeton.edu", :auto)], tunnel=true, topology=:master_worker)
    println(nprocs(), " processes")
    results = Results("inference")
end


@everywhere begin
    include("inference.jl")
    const SAMPLE_TIME = 100
    const N_PARTICLE = 2000
    const N_LATIN = 400
    const N_BO = 400

    const RETEST = false
    const N_PARAM = 6
    const REWEIGHT = false
    const data = Datum.(trials, SAMPLE_TIME)
end
# %% ====================  ====================


if get(ARGS, 1, "") == "worker"
    start_worker()
elseif get(ARGS, 1, "") == "master"
    start_master(wait=false)

    space = Box(
        :α => (10, 100, :log),
        :obs_sigma => (1, 60),
        :sample_cost => (0.0004, 0.002, :log),
        # :sample_cost => (1e-5, 1e-2, :log),
        :switch_cost => (1, 60),
        :µ => (0, 2 * μ_emp),
        :σ => (σ_emp / 4, 4 * σ_emp),
    )
    save(results, :space, space)

    const RAND_LOGP = sum(rand_logp.(data))
    const MAX_LOSS = 10

    function plogp(prm, particles=N_PARTICLE)
        smap(eachindex(data)) do i
            logp(prm, data[i], particles)
        end |> sum
    end

    function loss(x, particles=N_PARTICLE)
        prm = Params(;space(x)...)
        min(MAX_LOSS, plogp(prm, particles) / RAND_LOGP)
    end

    println("Begin GP minimize")
    @timed opt, t = gp_minimize(loss, N_PARAM, N_LATIN, N_BO; file="$results/opt_xy")
    res = (
        Xi = collect.(opt.Xi),
        yi = opt.yi,
        emin = expected_minimum(opt),
        runtime = t,
    )
    save(results, :opt, res)

    # %% ==================== Check top 20 to find best ====================
    if RETEST
        println("Computing loss for top 20.")
        ranked = sortperm(res.yi)
        top20 = res.Xi[ranked[1:20]]
        top20_loss = map(top20) do x
            fx, elapsed = @timed loss(x, 10 * N_PARTICLE)
            println(round.(x; digits=3),
                    " => ", round(fx; digits=4),
                    "   ", round(elapsed; digits=1), " seconds")
            fx
        end
        best = top20[argmin(top20_loss)]
    else
        best = res.Xi[argmin(res.yi)]
    end

    prm = Params(;space(best)...)
    println("MLE ", prm)

    lp = plogp(prm, 10000)
    println("Log Likelihood: ", lp)
    # println("BIC: ", log(N_OBS) * N_PARAM - 2 * lp)
    # println("AIC: ", 2 * N_PARAM - 2 * lp)

    save(results, :mle, (
        policy=SoftBlinkered(prm),
        prior=(prm.µ, prm.σ),
        logp=lp
    ))


    # %% ==================== Examine loss function around minimum ====================
    using JSON

    println("Explore loss function near discovered minimum.")
    diffs = -0.1:0.02:0.1

    cross = map(1:N_PARAM) do i
        map(diffs) do d
            x = copy(best)
            x[i] += d
            try
                loss(x)
            catch
                NaN
            end
        end
    end

    open("$results/cross.json", "w+") do f
        write(f, json(cross))
    end

    cross = map(1:N_PARAM) do i
        map(0:0.1:1) do d
            x = copy(best)
            x[i] = d
            try
                loss(x)
            catch
                NaN
            end
        end
    end

    open("$results/cross_full.json", "w+") do f
        write(f, json(cross))
    end

end

