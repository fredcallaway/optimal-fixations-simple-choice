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
    p_switch::Union{Missing,Float64}
end
BMPSPolicy(m::MetaMDP, θ::Vector{Float64}, α=Inf, p_switch=missing) = BMPSPolicy(m, BMPSWeights(θ), float(α), p_switch)

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

"Full VOC"
function voc(pol::BMPSPolicy, b::Belief)
    fast_voc(pol, b) .+ pol.θ.vpi * vpi(b)
end

"Selects a computation to take in the given belief state"
function act(pol::BMPSPolicy, b::Belief; clever=true)
    if !ismissing(pol.p_switch)
        # use the lesioned version of the model that chooses where to fixate randomly
        return act_lesioned(pol, b)
    end
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

function act_lesioned(pol::BMPSPolicy, b::Belief)
    θ = pol.θ
    @assert pol.α < Inf
    @assert !ismissing(pol.p_switch)
    @assert pol.m.switch_cost ≈ 0.

    cs = 1:pol.m.n_arm
    # probability that each computation will be executed if we don't terminate
    probs = map(cs) do c
        if b.focused == 0
            1 / length(cs)
        elseif c == b.focused
            1 - pol.p_switch
        else 
            pol.p_switch / (length(cs) - 1)
        end
    end
    @assert sum(probs) ≈ 1

    # Compute expected value of computation for continuing
    voi1_ = sum(p * voi1(b, c) for (c, p) in zip(cs, probs))
    voi_action_ = sum(p * voi_action(b, c) for (c, p) in zip(cs, probs))
    voc = voi1_ + voi_action_ - pol.m.sample_cost - θ.cost  # assume no switch cost

    # Add Gumbel noise to imitate softmax
    voc += rand(G) / pol.α  # for continue action
    voc -= rand(G) / pol.α  # for term action

    # Try putting VPI weight on VOI_action (a lower bound on VPI)
    voc + θ.vpi * voi_action_ > 0 && return sample(cs, Weights(probs))

    θ.vpi == 0. && return ⊥  # no weight on VPI, VOC can't improve

    # Try actual VPI.
    voc + θ.vpi * vpi(b) > 0 && return sample(cs, Weights(probs))

    # Nope.
    return ⊥
end



# ---------- Optimization helpers ---------- #

"Identifies the cost parameter that makes a hard-maximizing policy never take any computations."
function max_cost(m::MetaMDP)
    θ = [1., 0, 0, 1]
    b = Belief(m)
    # s = State(m)
    # b = Belief(s)
    function computes()
        pol = BMPSPolicy(m, θ)
        all(pol(b) != ⊥ for i in 1:30)
    end

    while computes()
        θ[1] *= 2
    end

    while !computes()
        θ[1] /= 2
        if θ[1] < 2^-10
            error("Computation is too expensive")
        end
    end

    step_size = θ[1] / 10
    while computes()
        θ[1] += step_size
    end
    θ[1]
end

"Transforms a value from the 3D unit hybercube to weights for BMPS"
function x2theta(mc, x)
    # This is a trick to go from two Uniform(0,1) samples to 
    # a unifomrm sample in the 3D simplex.
    voi_weights = diff([0; sort(collect(x[2:3])); 1])
    [x[1] * mc; voi_weights]
end



 