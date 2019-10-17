using Memoize
using Random
using Distributions
using StatsBase
# using OnlineStats
using QuadGK

include("utils.jl")
include("voi.jl")

BMPSWeights = NamedTuple{(:cost, :voi1, :voi_action, :vpi),Tuple{Float64,Float64,Float64,Float64}}
"A metalevel policy that uses the BMPS features"
struct BMPSPolicy <: Policy
    m::MetaMDP
    θ::BMPSWeights
    α::Float64
end
BMPSPolicy(m::MetaMDP, θ::Vector{Float64}, α=Inf) = BMPSPolicy(m, BMPSWeights(θ), float(α))

"Selects a computation to perform in a given belief."
(pol::BMPSPolicy)(b::Belief) = act(pol, b)

"VOC without VPI feature"
function fast_voc(pol::BMPSPolicy, b::Belief)
    θ = pol.θ
    map(1:pol.m.n_arm) do c
        -cost(pol.m, b, c) +
        -θ.cost +
        θ.voi1 * voi1(b, c) +
        θ.voi_action * voi_action(b, c)
    end
end


function full_voc(pol::BMPSPolicy, b::Belief, vpi_samples=10000)
    fast_voc(pol, b) .+ pol.θ.vpi * vpi(b, vpi_samples)[1]
end


function act(pol::BMPSPolicy, b::Belief; clever=true)
    if !clever && (pol.α < Inf)
        voc = [0; full_voc(pol, b)]
        p = softmax(pol.α .* voc)
        return sample(0:pol.m.n_arm, Weights(p))
    end

    θ = pol.θ
    voc = fast_voc(pol, b)
    if pol.α < Inf
        voc .+= rand(Gumbel(), pol.m.n_arm) ./ pol.α
        voc .-= rand(Gumbel()) / pol.α  # for term action
    else
        # break ties randomly
        voc .+= 1e-10 * rand(length(voc))
    end

    # Choose candidate based
    v, c = findmax(voc)
    v > 0 && return c

    # No computation is good enough without VPI.
    # Try putting VPI weight on VOI_action (a lower bound on VPI)
    v + θ.vpi * voi_action(b, c) > 0 && return c

    θ.vpi == 0. && return ⊥  # no weight on VPI, VOC can't improve

    # Try actual VPI.
    v + θ.vpi * vpi_clever(b) > 0 && return c

    # Nope.
    return ⊥
end


