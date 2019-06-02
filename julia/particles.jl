using Distributions
using Statistics
using Printf

# %% ==================== Random walk with observed sign ====================
x0 = 0.
transition(x, o) = x + randn()
obs_likelihood(x, o) = float(Int(x > 0) == o)

function gen_obs(n)
    x = x0
    obs = Int[]
    for i in 1:n
        push!(obs, Int(x > 0))
        x = transition(x, 1)
    end
    obs
end

n_obs = 4
obs = [0, 0, 0, 1]
mc_lik = mean(gen_obs(n_obs) == obs for i in 1:100000)

n_particle = 100000
x = [x0 for _ in 1:n_particle]
x1 = copy(x)
w0 = ones(n_particle) ./ n_particle
w = copy(w0)
η = 1

function resample!()
    n_offspring = rand(Multinomial(n_particle, w))
    j = 1
    for i in 1:n_particle
        for _ in 1:n_offspring[i]
            x1[j] = x[i]
            j += 1
        end
    end
    x .= x1
    w .= w0
end

for o in obs
    for i in 1:n_particle
        w[i] *= obs_likelihood(x[i], o)
        x[i] = transition(x[i], o)
    end
    sumw = sum(w)
    η *= sumw
    w ./= sumw
    resample!()
end

# 1 / sum(w .^ 2)
# sum(w .> 0)
@printf "%.3f  %.3f\n" mc_lik η

# %% ====================  ====================

particles = [Particle(0.) for _ in 1:n]

history = Vector{Particle}[]
pf = ParticleFilter(particles, transition, obs_likelihood)
run!(pf, obs, callback=ps->push!(history, deepcopy(ps)))
sort!(pf.particles, by=p->-p.w)


# # %% ====================  ====================
# using StatPlots
#
# density([p.x for p in history[1]])
# plot([h[1].x for h in history])
# plot([h[1].w for h in history])
# plot([mean(p.w for p in h) for h in history])
#
#
#
#
# # %% ==================== Sorting the particles ====================
#
# using DataStructures
#
# sd = SortedMultiDict()
#
# push!(sd, 1=>"a")
# push!(sd, 2=>"a")
#
# h = binary_maxheap(Particle.(1:10))
# x1 = pop!(h)
# x2 = pop!(h)
#
# x1.x = x2.x
# x1.w = x2.w = x2.w / 2
# push!(h, x1)
# push!(h, x2)
#
#
# import Base
# Base.isless(p1::Particle, p2::Particle) = p1.w < p2.w
#
# # %% ==================== Particle Library ====================
#
# dynamics(x, u, rng) = transition(x)
# observation(x_previous, u, x, o) = obs_likelihood(o, x)
#
# model = ParticleFilterModel{Float64}(dynamics, observation)
# pf = SIRParticleFilter(model, 10)
#
# b = ParticleCollection([1.0, 2.0, 3.0, 4.0])
# u = 1.0
# b_new = update(pf, b, u, y)
# b = update(pf, b, 0, 1)



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

# %% ==================== Metagreedy ====================
include("model.jl")
m = MetaMDP(n_arm=2, sample_cost=1e-4, obs_sigma=5)
pol = MetaGreedy(m)

s = State(m)
function sample_fixations()
    fix = Int[]
    rollout(m, pol, state=s, callback=(b,c)->push!(fix, c))
    fix
end

# obs = sample_fixations()
obs = [2, 2, 0]
println(obs)
b = Belief(s)
function transition(b, c)
    b = deepcopy(b)
    step!(m, b, s, c)
    b
end
obs_likelihood(b, c) = Int(pol(b) == c)

function lik()
    particles = [Particle(b) for i in 1:1000]
    pf = ParticleFilter(particles, transition, obs_likelihood)
    run!(pf, obs)
    mean(p.w for p in particles)
end

lik()
# %% ====================  ====================
lik2() = mean(sample_fixations() == obs for i in 1:1000)
l1 = [lik() for i in 1:100]
l2 = [lik2() for i in 1:100]
mean(l1)
mean(l2)
std(l1)
std(l2)

# %% ==================== Coin flipping ====================

x0 = []
transition(x) = [x..., Int(rand() > 0.5)]
obs_likelihood(o, x) = x[end] == 0
gen_obs(n) = rand(0:1, n)

N = 3
obs = gen_obs(N)

#=
def estimate_loglikelihood(human_response)
	L = 0
	n = 1

	while True:
		model_sample = sample_from_model()
		if model_sample == human_response:
			break
		else:
			L = L + 1/n
			n = n + 1
	return L
=#
function bas_logp()
    L = 0; n = 1
    while true
        if gen_obs(N) == obs
            return -L
        end
        L += 1/n
        n += 1
    end
end

mean(exp(bas_logp()) for i in 1:100000)
exp(mean(bas_logp() for i in 1:100000))

# function lik()
#     particles = [Particle([]) for i in 1:1000]
#     pf = ParticleFilter(particles, transition, obs_likelihood)
#     run!(pf, obs)
#     mean(p.w for p in particles)
# end


function lik2()
    mean(gen_obs(n) == obs for i in 1:1000)
end

l1 = [lik() for i in 1:100]
l2 = [lik2() for i in 1:100]
ltrue = 0.5^n
mae(x, y) = mean(abs.(x .- y))
mae(l1, ltrue)
mae(l2, ltrue)
