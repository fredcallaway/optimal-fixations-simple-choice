using Memoize
using Random
using Distributions
using StatsBase
# using OnlineStats

include("utils.jl")
include("voi.jl")

const G = Gumbel()

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

function voc(pol::BMPSPolicy, b::Belief)
    fast_voc(pol, b) .+ pol.θ.vpi * vpi(b)
end

function act(pol::BMPSPolicy, b::Belief; clever=true)
    θ = pol.θ
    voc = fast_voc(pol, b)

    if !clever  # computationally inefficient, but clearly correct
        voc .+= θ.vpi * vpi(b)
        if pol.α == Inf
            v, c = findmax(voc)
            return (v > 0) ? c : ⊥
        else
            p = softmax(pol.α .* [0; voc])
            return sample(0:pol.m.n_arm, Weights(p))
        end
    end

    if pol.α < Inf
        # gumbel-max trick
        voc .+= rand(G, pol.m.n_arm) ./ pol.α
        voc .-= rand(G) / pol.α  # for term action
    else
        # break ties randomly
        voc .+= 1e-10 * rand(length(voc))
    end

    # Choose candidate based on cheap voc
    v, c = findmax(voc)

    # Try putting VPI weight on VOI_action (a lower bound on VPI)
    v + θ.vpi * voi_action(b, c) > 0 && return c

    θ.vpi == 0. && return ⊥  # no weight on VPI, VOC can't improve

    # Try actual VPI.
    v + θ.vpi * vpi(b) > 0 && return c

    # Nope.
    return ⊥
end


