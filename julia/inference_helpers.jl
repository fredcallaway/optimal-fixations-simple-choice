include("utils.jl")
include("model.jl")
include("SIR.jl")
# include("ParticleFilters.jl")
# include("job.jl")
# include("human.jl")
# include("simulations.jl")

function ParticleFilter(policy, ε::Float64, value::Vector{Float64})
    m = policy.m
    s = State(m, value)
    init() = Belief(s)
    transition(b, c) = begin
        step!(m, b, s, c)
        b
    end
    obs_p(b, c) = begin
        prediction = rand() < ε ? rand(0:m.n_arm) : policy(b)
        Int(prediction == c)
    end
    ParticleFilter(init, transition, obs_p)
end

# function old_logp(policy, ε, value, samples, choice; n_particle=10000)
#     pf = ParticleFilter(policy, ε, value)
#     ps = run!(pf, [samples; 0]; n_particle=n_particle)
#     for p in ps
#         # choice probability
#         p.w *= softmax(1e10 * p.x.mu)[choice]
#     end
#     log(mean(p.w for p in ps))
# end

function logp(policy, value, samples, choice, n_particle=10000)
    m = policy.m
    s = State(m, value)
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
    x, w = sir.P.x, sir.P.w
    for i in 1:length(x)
        w[i] *= softmax(policy.α * x[i].mu)[choice]
    end
    lp += log(reweight!(sir.P))
    lp
end

function rand_logp(t::Trial)
    samples = discretize_fixations(t, sample_time=100)
    log(1/4) * (length(samples) + 1) + log(1/3)
end
rand_logp(tt::Table{Trial}) = sum(rand_logp.(tt))

function stay_logp(t::Trial, ε)
    samples = discretize_fixations(t, sample_time=100)
    switch = diff(samples) .!= 0
    p_first = log(1/3)
    p_stay = log(1-ε) * sum(.!switch)
    p_switch = log(ε * 1/4) * sum(switch)
    p_end = log(ε * 1/4)
    return p_first + p_stay + p_switch + p_end
end
stay_logp(tt::Table{Trial}, ε) = sum(stay_logp.(tt, ε))