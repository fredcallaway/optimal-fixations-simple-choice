
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


# %% ==================== Random walk with observed sign ====================
x0 = 0
transition(x) = x + randn()
obs_likelihood(o, x) = float(Int(x > 0) == o)

function gen_obs(n)
    x = x0
    obs = Int[]
    for i in 1:n
        x = transition(x)
        push!(obs, act(x))
    end
    obs
end

println(gen_obs(4))
n = 4
obs = [0, 1, 1, 1]
mean(gen_obs(n) == obs for i in 1:1000000)
# %%
particles = [Particle(0.) for _ in 1:n]

history = Vector{Particle}[]
pf = ParticleFilter(particles, transition, obs_likelihood)
run!(pf, obs, callback=ps->push!(history, deepcopy(ps)))
sort!(pf.particles, by=p->-p.w)
#
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
