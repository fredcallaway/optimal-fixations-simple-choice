# %%
import CSV
using TypedTables
using SplitApplyCombine
using Distributions
using StatsBase
using Lazy: @>>

# %%

valmap(f, d::Dict) = Dict(k => f(v) for (k, v) in d)
keymap(f, d::Dict) = Dict(f(k) => v for (k, v) in d)

const data = Table(CSV.File("../krajbich_PNAS_2011/data.csv"; allowmissing=:none));

function reduce_trial(t::Table)
    r = t[1]
    (choice = argmax([r.choice1, r.choice2, r.choice3]),
     value = Float64[r.rating1, r.rating2, r.rating3],
     subject = r.subject,
     trial = r.trial,
     rt = r.rt,
     fixations = combinedims([t.leftroi, t.middleroi, t.rightroi]) * [1, 2, 3],
     fix_times = t.eventduration)
end

function normalize_values!(trials)
    for (subj, g) in group(x->x.subject, trials)
        μ, σ = juxt(mean, std)(flatten(g.value))
        for v in g.value
            v .-= μ
            v ./= σ
        end
    end
    return trials
end

const trials = @>> begin
    data
    group(x->(x.subject, x.trial))
    values
    map(reduce_trial)
    Table
    normalize_values!
end


function parse_computations(t; sample_time=50)
    cs = mapmany(t.fixations, t.fix_times) do item, ft
        repeat([item], Int(round(ft/sample_time)))
    end
    push!(cs, 0)
    (cs=cs, choice=t.choice)
end

# %% ==================== Explore data ====================

num_fixation(trials[1].fixations)
trials[2].fixations

map(typeof, group(x->x.subject, trials))

@>> begin
    trials
    group(x->x.subject)
    values
    map
end


groupsum(x->x.subject, x->length(x.fixations), trials)
groupsum(x->x.subject, x->x.rt, trials)


# %% ==================== Simulate ====================
include("model.jl")
function simulate(m::MetaMDP, value)
    cs = Int[]
    s = State(m, value)
    choice = rollout(m, MetaGreedy(m), state=s, callback=(b,c)->push!(cs, c)).choice
    (cs=cs, choice=choice)
end


num_fixation(cs) = sum(diff(cs) .!= 0)
m = MetaMDP(obs_sigma=5, sample_cost=0.001, switch_cost=5)
x = simulate(m, trials[1].value)




# %% ==================== Particle filter ====================
include("model.jl")
include("ParticleFilters.jl")

m = MetaMDP(obs_sigma=5, sample_cost=0.001, switch_cost=5)

function simulate(m, s)
    cs = Int[]
    choice = rollout(m, MetaGreedy(m), state=s, callback=(b,c)->push!(cs, c)).choice
    (cs=cs, choice=choice)
end

function choice_probs(b::Belief)
    is_best = b.mu .== maximum(b.mu)
    is_best / sum(is_best)
end

function ParticleFilter(m::MetaMDP, t)
    pol = MetaGreedy(m)
    s = State(m, t.value)
    init() = Belief(s)
    transition(b, c) = begin
        step!(m, b, s, c)
        b
    end
    obs_p(b, c) = Int(pol(b) == c)
    ParticleFilter(init, transition, obs_p)
end

function logp(pf::ParticleFilter, d; n_particle=10000)
    ps = run!(pf, d.cs; n_particle=n_particle)
    for p in ps
        # choice probability
        p.w *= softmax(1e10 * p.x.mu)[d.choice]
    end
    log(mean(p.w for p in ps))
end

function logp(m::MetaMDP, t; sample_time=500, n_particle=10000)
    d = parse_computations(t; sample_time=sample_time)
    pf = ParticleFilter(m, t)
    logp(pf, d; n_particle=n_particle)
end


# %% ====================  ====================
m1 = MetaMDP(obs_sigma=3)
m2 = MetaMDP(obs_sigma=5)
d = simulate(m1, State(m1, t.value))
p1 = logp(ParticleFilter(m1, t), d)
p2 = logp(ParticleFilter(m2, t), d)
(p1, p2)


# %% ====================  ====================
t = trials[1000 .<= trials.rt .<= 5000][1]
s = State(m, t.value)

steps, choices = repeatedly(10) do
    roll = rollout(m, MetaGreedy(m), state=s)
    (roll.steps, roll.choice)
end |> invert
mean(steps)
proportions(choices, 3) |> Tuple


pf = ParticleFilter(m, t)
d = parse_computations(t; sample_time=100)
history = []
ps = run!(pf, d.cs, callback=(particles->push!(history, deepcopy(particles))))
# %% ====================  ====================
function plot_particles(i)
    h = history[i]
    c = d.cs[i]
    mu = combinedims([p.x.mu for p in h])
    w = [p.w for p in h]
    w ./= maximum(w)
    plot(mu, label="", color=:black, α=0.1w', ylim=(-1, 1), title=string(c))
end
plots = [plot_particles(i) for i in 1:length(history)]
plot(plots...)

plot(1:3, title=string(1))
# %% ====================  ====================



cs = parse_computations(t; sample_time=200)
x = counts(cs)[2:end]
x / sum(x)

groupreduce(x->x[1], x->x[2], +, zip(t.fixations, t.fix_times))
