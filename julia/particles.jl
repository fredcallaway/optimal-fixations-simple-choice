using ParticleFilters, Distributions

dynamics(x, u, rng) = transition(x)
observation(x_previous, u, x, o) = 1





model = ParticleFilterModel{Float64}(dynamics, observation)
pf = SIRParticleFilter(model, 10)

b = ParticleCollection([1.0, 2.0, 3.0, 4.0])
u = 1.0
y = 3.0

b_new = update(pf, b, u, y)


# %% ====================  ====================
x0 = 0
act(x) = Int(x > 0)
transition(x) = 0.9x + 0.1randn()
obs_likelihood(o, x) = float(act(x) == o)

function gen_obs(n)
    x = x0
    obs = []
    for i in 1:n
        x = transition(x)
        push!(obs, act(x))
    end
    obs
end
gen_obs(10)
# %% ====================  ====================
n = 10
weights = ones(n) ./ n

using SplitApplyCombine
using Base.Iterators: cycle, take

mutable struct Particle{T} where T
    w::Float64
    x::T
end

function resample!(particles)
    g = group(p->p.w > 0, particles)
    haskey(g, false) || return  # no dead particles
    haskey(g, true) || return # uh oh!

    for (dead, living) in zip(g[false], cycle(g[true]))
        dead.x = living.x
        dead.w = living.w = living.w / 2
    end
end

function reweight!(particles, o)
    for p in particles
        p.x = transition(p.x)
        p.w = obs_likelihood(o, p.x)
    end
end

function observe!(particles, obs)
    for o in obs
        reweight!(particles, o)
        resample!(particles)
    end
end
# %% ====================  ====================
n = 100
# obs = gen_obs(40)

particles = [Particle(1, 0.) for _ in 1:n]
observe!(particles, obs)
mean(p.w for p in particles)
# %% ====================  ====================
for o in obs
    particles = transition.(particles)
    float(act.(particles) .== o)
end
