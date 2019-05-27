# %%
include("human.jl")
include("blinkered.jl")
include("elastic.jl")
include("inference_helpers.jl")
include("skopt.jl")


struct Datum
    value::Vector{Float64}
    samples::Vector{Int}
    choice::Int
end
Datum(t::Trial) = Datum(
    t.value,
    discretize_fixations(t),
    t.choice
)

struct Params
    α::Float64
    obs_sigma::Float64
    sample_cost::Float64
    switch_cost::Float64
    µ::Float64
    σ::Float64
end

rescale(x, low, high) = low + x * (high-low)
Params(x::Vector{Float64}) = Params(
    10 ^ rescale(x[1], 0, 2),
    rescale(x[2], 1, 10),
    10 ^ rescale(x[3], -4, -2),
    rescale(x[4], 1, 20),
    rescale(x[5], 0., µ_emp),
    σ_emp
)

MetaMDP(prm::Params) = MetaMDP(
    3,
    prm.obs_sigma,
    prm.sample_cost,
    prm.switch_cost,
)
SoftBlinkered(prm::Params) = SoftBlinkered(MetaMDP(prm), prm.α)
SoftBlinkered(Params(rand(5)))
value(prm::Params, d::Datum) = (d.value .- prm.µ) ./ prm.σ

function logp(prm::Params, d::Datum, particles=1000)
    policy = SoftBlinkered(prm)
    logp(policy, value(prm, d), d.samples, d.choice, particles)
end

function logp(prm::Params, dd::Vector{Datum}, particles=1000)
    map(dd) do d
        logp(prm, d, particles)
    end |> sum
end
const data = Datum.(trials)

const N_OBS = sum(length(d.samples) + 1 for d in data)
const RAND_LOSS = sum(rand_logp.(trials)) / N_OBS
const MAX_LOSS = -10 * RAND_LOSS

# %% ====================  ====================
if get(ARGS, 1, "") == "worker"
    start_worker()
else
    start_master()

    function plogp(prm, particles=1000)
        smap(eachindex(data)) do i
            logp(prm, data[i], particles)
        end |> sum
        # @distributed (+) for i in eachindex(data)
        #     logp(prm, data[i], particles)
        # end
    end

    function loss(x, particles=1000)
        prm = Params(x)
        min(MAX_LOSS, -plogp(prm, particles) / N_OBS)
    end


    # println("Begin GP minimize")
    # @time res = gp_minimize(loss, 5, 500, 200)

    # open("tmp/blinkered_opt", "w+") do f
    #     serialize(f, (
    #         Xi = collect.(res.Xi),
    #         yi = res.yi,
    #         emin = expected_minimum(res)
    #     ))
    # end

    res = open(deserialize, "tmp/blinkered_opt")


    # %% ==================== Check top 10 to find best ====================
    ranked = sortperm(res.yi)
    top20 = res.Xi[ranked[1:20]]
    top20_loss = map(top20) do x
        loss(x, 10000)
    end
    println("Top 20")
    println(top20_loss)
    best = top20[argmin(top20_loss)]

    # %% ==================== Examine loss function around minimum ====================
    diffs = -0.2:0.02:0.2

    cross = map(1:5) do i
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

    using JSON
    open("tmp/cross5.json", "w+") do f
        write(f, json(cross))
    end
    using Serialization
    open("tmp/blinkered_policy.jls", "w+") do f
        serialize(f, SoftBlinkered(Params(best)))
    end
end

