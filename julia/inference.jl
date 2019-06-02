include("elastic.jl")

if get(ARGS, 1, "") == "master"
    addprocs(topology=:master_worker)
    # addprocs([("griffiths-gpu01.pni.princeton.edu", :auto)], tunnel=true, topology=:master_worker)
    println(nprocs(), " processes")
end

# necessary for @with_kw macro below
@everywhere using Parameters

@everywhere begin
    include("human.jl")
    include("blinkered.jl")
    include("inference_helpers.jl")
    include("skopt.jl")

    using Serialization
    using JSON

    const N_PARTICLE = 500
    const N_LATIN = 200
    const N_BO = 100
    const SAMPLE_TIME = 100

    const OPTIMIZE = true
    const RETEST = true
    const N_PARAM = 5

    struct Datum
        value::Vector{Float64}
        samples::Vector{Int}
        choice::Int
    end
    Datum(t::Trial) = Datum(
        t.value,
        discretize_fixations(t; sample_time=SAMPLE_TIME),
        t.choice
    )

    @with_kw struct Params
        α::Float64
        obs_sigma::Float64
        sample_cost::Float64
        switch_cost::Float64
        µ::Float64
        σ::Float64
    end

    rescale(x, low, high) = low + x * (high-low)
    Params(x::Vector{Float64}) = Params(
        10 ^ rescale(x[1], 1, 2),
        rescale(x[2], 1, 60),
        10 ^ rescale(x[3], -5, -2),
        rescale(x[4], 10, 60),
        µ_emp * rescale(x[5], 0., 2),
        N_PARAM == 6 ? (σ_emp * 2 ^ rescale(x[6], -2, 2)) : σ_emp
    )

    MetaMDP(prm::Params) = MetaMDP(
        3,
        prm.obs_sigma,
        prm.sample_cost,
        prm.switch_cost,
    )
    SoftBlinkered(prm::Params) = SoftBlinkered(MetaMDP(prm), prm.α)
    value(prm::Params, d::Datum) = (d.value .- prm.µ) ./ prm.σ

    function logp(prm::Params, d::Datum, particles=N_PARTICLE)
        policy = SoftBlinkered(prm)
        logp(policy, value(prm, d), d.samples, d.choice, particles)
    end

    function logp(prm::Params, dd::Vector{Datum}, particles=N_PARTICLE)
        map(dd) do d
            logp(prm, d, particles)
        end |> sum
    end
    const data = Datum.(trials)

    const N_OBS = sum(length(d.samples) + 1 for d in data)
    const RAND_LOSS = sum(rand_logp.(trials)) / N_OBS
    const MAX_LOSS = -10 * RAND_LOSS
end
# %% ====================  ====================

if get(ARGS, 1, "") == "worker"
    start_worker()
elseif get(ARGS, 1, "") == "master"
    start_master(wait=false)

    using Dates
    timestamp = replace(split(string(now()), ".")[1], ':' => '-')
    results = "results/$timestamp"
    mkdir(results)
    println("Saving results to $results/")

    function plogp(prm, particles=N_PARTICLE)
        smap(eachindex(data)) do i
            logp(prm, data[i], particles)
        end |> sum
        # @distributed (+) for i in eachindex(data)
        #     logp(prm, data[i], particles)
        # end
    end

    function loss(x, particles=N_PARTICLE)
        prm = Params(x)
        min(MAX_LOSS, -plogp(prm, particles) / N_OBS)
    end

    # res = open(deserialize, "results/opt_xy")
    # loss(res.Xi[argmin(res.yi)])

    if OPTIMIZE
        println("Begin GP minimize")
        @time res = gp_minimize(loss, N_PARAM, N_LATIN, N_BO; file="$results/opt_xy")
        res = (
            Xi = collect.(res.Xi),
            yi = res.yi,
            emin = expected_minimum(res)
        )
        open("$results/blinkered_opt", "w+") do f
            serialize(f, res)
        end
    else
        res = open(deserialize, "$results/opt_xy")
    end

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

    prm = Params(best)
    println("MLE ", prm)
    open("$results/blinkered_policy.jls", "w+") do f
        serialize(f, (
            policy=SoftBlinkered(prm),
            prior=(prm.µ, prm.σ)
        ))
    end

    lp = plogp(best, 100000)
    println("Log Likelihood: ", lp)
    println("BIC: ", log(N_OBS) * N_PARAM - 2 * lp)
    println("AIC: ", 2 * N_PARAM - 2 * lp)

    # %% ==================== Examine loss function around minimum ====================
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

end

