using Memoize
using Random
using Distributions
using StatsBase
using OnlineStats
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


function expected_max_norm(μ, λ)
    dists = Normal.(μ, λ.^-0.5)
    mcdf(x) = mapreduce(*, dists) do d
        cdf(d, x)
    end

    quadgk(x->1-mcdf(x), 0, 5, atol=1e-5)[1] -
      quadgk(mcdf, -5, 0, atol=1e-5)[1]
end

function vpi_clever(b)
    expected_max_norm(b.μ, b.λ) - maximum(b.μ)
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
    end

    v, c = findmax(noisy(voc))
    v > 0 && return c

    # No computation is good enough without VPI.
    # Try putting VPI weight on VOI_action (a lower bound on VPI)
    v + θ.vpi * voi_action(b, c) > 0 && return c

    # Still no luck. Try actual VPI. To make the VPI estimation as fast and accurate as possible,
    # we don't use a fixed number of samples. Instead, we continually add samples until
    # the estimate is precise "enough". The estimate is precise enough when:
    # (1) the uncertainty point (VOC=0) is not within 3 standard errors, OR
    # (2) the standard error is less than 1e-4
    # (3) 100,000 samples of the VPI have been taken

    θ.vpi == 0. && return ⊥  # no weight on VPI, VOC can't improve

    v + θ.vpi * vpi_clever(b) > 0 && return c
    return ⊥

    vpi = Variance()  # tracks mean and variance of running VPI samples

    for i in 1:200
        fit!(vpi, vpi!(mem_zeros(500), b))  # add 500 samples
        μ_voc = v + θ.vpi * vpi.μ
        σ_voc = θ.vpi * √(vpi.σ2 / vpi.n)
        # (σ_voc < 1e-4 || abs(μ_voc - 0) > 3 * σ_voc) && println("$i iterations")
        (σ_voc < 1e-4 || abs(μ_voc - 0) > 3 * σ_voc) && break
        # (i == 100000) && println("Warning: VPI estimation did not converge.")
    end
    v + θ.vpi * vpi.μ > 0 && return c
    return ⊥
end


