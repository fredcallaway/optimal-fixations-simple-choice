
using Distributions
struct Particles{T}
    x::Vector{T}
    x1::Vector{T}
    w::Vector{Float64}
    nc::Vector{Int}
end

function Particles{T}(n::Int) where T
    Particles(
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Int}(undef, n),
    )
end

struct SIR{I,T,L,P}
    init::I
    transition::T
    likelihood::L
    P::Particles{P}
end

# %% ====================  ====================
SIR(I::Function, T::Function, L::Function, P::Type, n::Int) = SIR(I, T, L, Particles{P}(n::Int))

function resample!(P::Particles)
    x, x1, w, nc = P.x, P.x1, P.w, P.nc
    Distributions.rand!(Multinomial(length(x), w), nc)
    j = 1
    for i in 1:length(x)
        for _ in 1:nc[i]
            x1[j] = copy(x[i])
            j += 1
        end
    end
    x .= x1
    w .= 1. / length(x)
end

function reset!(s::SIR)
    x, w = s.P.x, s.P.w
    w0 = 1. / length(x)
    for i in 1:length(x)
        x[i] = s.init()
        w[i] = w0
    end
end

function step!(s::SIR, obs)
    x, w = s.P.x, s.P.w
    for i in 1:length(x)
        w[i] *= s.likelihood(x[i], obs)
        x[i] = s.transition(x[i], obs)
    end
end

function reweight!(s::SIR)
    w = s.P.w
    sumw = sum(w)
    w ./= sumw
    return sumw
end

function logp(s::SIR, observations)
    reset!(s)
    logp = 0.
    for obs in observations
        step!(s, obs)
        logp += log(reweight!(s))
        resample!(s.P)
    end
    logp
end
