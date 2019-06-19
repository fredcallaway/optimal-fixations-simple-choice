using Parameters
using Serialization
using JSON

include("utils.jl")
include("model.jl")
include("SIR.jl")
include("human.jl")
include("blinkered.jl")
include("skopt.jl")

# %% ==================== Data ====================

struct Datum
    value::Vector{Float64}
    samples::Vector{Int}
    choice::Int
end
Datum(t::Trial, sample_time::Int) = Datum(
    t.value,
    discretize_fixations(t; sample_time=sample_time),
    t.choice
)

# %% ==================== Parameters ====================
@with_kw struct Params
    α::Float64
    obs_sigma::Float64
    sample_cost::Float64
    switch_cost::Float64
    µ::Float64
    σ::Float64
end

MetaMDP(prm::Params) = MetaMDP(
    3,
    prm.obs_sigma,
    prm.sample_cost,
    prm.switch_cost,
)
SoftBlinkered(prm::Params) = SoftBlinkered(MetaMDP(prm), prm.α)
value(prm::Params, d::Datum) = (d.value .- prm.µ) ./ prm.σ


# %% ==================== Likelihood ====================

function logp(policy, v, samples, choice, n_particle=10000; reweight=true)
    m = policy.m
    s = State(m, v)
    init() = Belief(s)
    transition(b, c) = begin
        step!(m, b, s, c)
        b
    end
    likelihood(b, c) = action_probs(policy, b)[c+1]  # +1 because TERM is at position 1

    sir = SIR(init, transition, likelihood, Belief, n_particle)
    lp = logp(sir, samples)

    # Termination and choice
    step!(sir, 0)
    lp += log(reweight!(sir.P))
    resample!(sir.P)

    x, w = sir.P.x, sir.P.w
    for i in 1:length(x)
        w[i] *= softmax(policy.α * x[i].mu)[choice]
    end
    if reweight
        sample_logp = lp / (length(samples) + 1)
        choice_logp = log(reweight!(sir.P))
        return sample_logp + choice_logp
    else
        return lp + log(reweight!(sir.P))
    end
end

function logp(prm::Params, d::Datum, particles=N_PARTICLE)
    policy = SoftBlinkered(prm)
    logp(policy, value(prm, d), d.samples, d.choice, particles;
         reweight=REWEIGHT)
end

function logp(prm::Params, dd::Vector{Datum}, particles=N_PARTICLE)
    mapreduce(+, dd) do d
        logp(prm, d, particles)
    end
end

function rand_logp(d::Datum)
    log(1/4) * (length(d.samples) + 1) + log(1/3)
end

function stay_logp(t::Trial, ε)
    samples = discretize_fixations(t, sample_time=SAMPLE_TIME)
    switch = diff(samples) .!= 0
    p_first = log(1/3)
    p_stay = log(1-ε) * sum(.!switch)
    p_switch = log(ε * 1/4) * sum(switch)
    p_end = log(ε * 1/4)
    return p_first + p_stay + p_switch + p_end
end
stay_logp(tt::Table{Trial}, ε) = sum(stay_logp.(tt, ε))

















