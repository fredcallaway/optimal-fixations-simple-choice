using SplitApplyCombine
using Base.Iterators: cycle, take
using Random: shuffle!

export ParticleFilter, run!


mutable struct Particle{T}
    w::Float64
    x::T
end
Particle(x) = Particle(1., x)


struct ParticleFilter{I, T, O}
    initialize::I
    transition::T
    obs_likelihood::O
end

function resample!(particles)
    g = group(p -> p.w > 0, particles)
    haskey(g, false) || return  # no dead particles
    haskey(g, true) || error("No more particles")
    shuffle!(g[true])

    for (dead, living) in zip(g[false], cycle(g[true]))
        dead.x = copy(living.x)
        dead.w = living.w = living.w / 2
    end
end



function update!(pf::ParticleFilter, particles, o)
    for p in particles
        p.w *= pf.obs_likelihood(p.x, o)
        p.x = pf.transition(p.x, o)
    end
end

function run!(pf::ParticleFilter, obs; n_particle=1000, callback=(particles->nothing))
    particles = [Particle(pf.initialize()) for i in 1:n_particle]
    callback(particles)
    for o in obs
        for p in particles
            p.w *= pf.obs_likelihood(p.x, o)
        end
        resample!(particles)
        for p in particles
            p.x = pf.transition(p.x, o)
        end
        callback(particles)
    end
    particles
end
