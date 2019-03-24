include("utils.jl")
include("model.jl")
include("ParticleFilters.jl")
include("job.jl")
include("human.jl")
include("simulations.jl")
include("SIR.jl")

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

function logp(policy, ε, value, samples, choice; n_particle=10000)
    pf = ParticleFilter(policy, ε, value)
    ps = run!(pf, [samples; 0]; n_particle=n_particle)
    for p in ps
        # choice probability
        p.w *= softmax(1e10 * p.x.mu)[choice]
    end
    log(mean(p.w for p in ps))
end

function sir_logp(policy, ε, value, samples, choice, n_particle=10000)
    m = policy.m
    s = State(m, value)
    init() = Belief(s)
    transition(b, c) = begin
        step!(m, b, s, c)
        b
    end
    likelihood(b, c) = (1 - ε) * (c == policy(b)) + ε/4

    sir = SIR(init, transition, likelihood, Belief, n_particle)
    lp = logp(sir, samples)

    # Termination and choice
    step!(sir, 0)
    x, w = sir.P.x, sir.P.w
    for i in 1:length(x)
        w[i] *= softmax(1e10 * x[i].mu)[choice]
    end
    lp += log(reweight!(sir.P))
    lp
end
