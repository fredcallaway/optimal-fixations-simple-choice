# %%
using Distributed
addprocs(48)

@everywhere include("inference_helpers.jl")
@everywhere include("human.jl")
@everywhere include("blinkered.jl")

include("skopt.jl")
using Distributions
using StatsBase

# %% ====================  ====================

@everywhere begin
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
end

# %% ==================== GP Optimization ====================

# function plogp(prm, trials, particles=5000)
#     @distributed (+) for t in trials
#         logp(prm, t, particles)
#     end
# end

function plogp(prm, particles=5000)
    @distributed (+) for i in eachindex(data)
        logp(prm, data[i], particles)
    end
end

N_OBS = sum(length(d.samples) + 1 for d in data)
rand_loss = sum(rand_logp.(trials)) / N_OBS
MAX_LOSS = -10 * rand_loss

function loss(x, particles=1000)
    prm = Params(x)
    min(MAX_LOSS, -plogp(prm, particles) / N_OBS)
end
@time res = gp_minimize(loss, 5, 500, 200)

# %% ==================== Check top 10 to find best ====================
ranked = sortperm(res.yi)
top10 = res.Xi[ranked[1:10]]
top10_loss = map(top10) do x
    loss(collect(x))
end
best = collect(top10[argmin(top10_loss)])

# %% ==================== Examine loss function around minimum ====================
diffs = -0.1:0.01:0.1

cross = map(1:5) do i
    map(diffs) do d
        x = copy(best)
        x[i] += d
        try
            -plogp(Params(x), trials)
        catch
            NaN
        end
    end
end

using JSON
open("tmp/cross3.json", "w+") do f
    write(f, json(cross))
end
using Serialization
open("tmp/blinkered_policy.jls", "w+") do f
    serialize(f, SoftBlinkered(Params(best)))
end
Params(best).µ
# %% ====================  ====================

function choice_probs(v, n=10000)
    s = State(pol.m, v)
    choices = zeros(3)
    for i in 1:n
        choices[rollout(pol, state=s).choice] += 1
    end
    choices / n
end

# %% ====================  ====================

loss(x, 10000)

optimize([1e-5], [1.], [0.99], Fminbox(BFGS()), autodiff=:forward) do x
    -stay_logp(trials, x[1])
end


# %% ====================  ====================

using Optim
stay_res = optimize([1e-5], [1.], [0.99], Fminbox(BFGS()), autodiff=:forward) do x
    -stay_logp(trials, x[1])
end
minimum(stay_res)


# %% ====================  ====================
function simulate(prm::Params, t::Trial)
    policy = SoftBlinkered(prm)
    s = State(policy.m, value(prm, t))
    cs = Int[]
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c); max_steps=1000)
    (samples=cs[1:end-1], choice=roll.choice, value=t.value)
end
prm = options[i]
sim = map(trials) do t
    simulate(prm, t)
end


rt = length.(sim.samples) .* 100

value_loss(t) = maximum(t.value) - t.value[t.choice]

mean(value_loss.(trials))
mean(value_loss.(sim))



# %% ====================  ====================

