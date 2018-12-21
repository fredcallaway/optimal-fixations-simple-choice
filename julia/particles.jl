using ParticleFilters, Distributions
using SplitApplyCombine
using Base.Iterators: cycle, take

mutable struct Particle{T}
    w::Float64
    x::T
end
Particle(x) = Particle(1., x)

struct ParticleFilter
    particles::Vector{Particle}
    transition
    obs_likelihood
end

using Random: shuffle!
function resample!(pf::ParticleFilter)
    g = group(p -> p.w > 0, pf.particles)
    haskey(g, false) || return  # no dead particles
    haskey(g, true) || error("No more particles")
    shuffle!(g[true])

    for (dead, living) in zip(g[false], cycle(g[true]))
        dead.x = living.x
        dead.w = living.w = living.w / 2
    end
end

function reweight!(pf::ParticleFilter, o)
    for p in pf.particles
        p.w *= obs_likelihood(p.x, o)
        p.x = transition(p.x, o)
    end
end

function run!(pf::ParticleFilter, obs; callback=(particles->nothing))
    for o in obs
        reweight!(pf, o)
        resample!(pf)
        callback(pf)
    end
end

# %% ==================== Metagreedy ====================
include("model.jl")
m = MetaMDP(n_arm=2)
pol = MetaGreedy(m)

s = State(m)
function sample_fixations()
    fix = Int[]
    rollout(m, pol, state=s, callback=(b,c)->push!(fix, c))
    fix
end

obs = sample_fixations()

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
lik2() = mean(sample_fixations() == obs for i in 1:1000)

l1 = [lik() for i in 1:100]
l2 = [lik2() for i in 1:100]
mean(l1)
mean(l2)
std(l1)
std(l2)
# # %% ==================== Coin flipping ====================
#
# x0 = []
# transition(x) = [x..., Int(rand() > 0.5)]
# obs_likelihood(o, x) = x[end] == 0
# gen_obs(n) = rand(0:1, n)
#
# n = 10
# obs = gen_obs(n)
#
# function lik()
#     particles = [Particle([]) for i in 1:1000]
#     pf = ParticleFilter(particles, transition, obs_likelihood)
#     run!(pf, obs)
#     mean(p.w for p in particles)
# end
#
#
# function lik2()
#     mean(gen_obs(n) == obs for i in 1:1000)
# end
#
# l1 = [lik() for i in 1:100]
# l2 = [lik2() for i in 1:100]
# ltrue = 0.5^n
# mae(x, y) = mean(abs.(x .- y))
# mae(l1, ltrue)
# mae(l2, ltrue)
#
#
# # %% ==================== Random walk with observed sign ====================
# x0 = 0
# act(x) = Int(x > 0)
# transition(x) = 0.9x + 0.1randn()
# obs_likelihood(o, x) = float(act(x) == o)
#
# function gen_obs(n)
#     x = x0
#     obs = Int[]
#     for i in 1:n
#         x = transition(x)
#         push!(obs, act(x))
#     end
#     obs
# end
#
# n = 100
# obs = gen_obs(40)
#
# particles = [Particle(0.) for _ in 1:n]
#
# history = Vector{Particle}[]
# pf = ParticleFilter(particles, transition, obs_likelihood)
# run!(pf, obs, callback=ps->push!(history, deepcopy(ps)))
# sort!(pf.particles, by=p->-p.w)
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
