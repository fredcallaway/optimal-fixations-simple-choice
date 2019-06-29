using Memoize
using Random
using Distributions

include("voi.jl")


BMPSWeights = NamedTuple{(:cost, :voi1, :voi_action, :vpi),Tuple{Float64,Float64,Float64,Float64}}
"A metalevel policy that uses the BMPS features"
struct BMPSPolicy <: Policy
    m::MetaMDP
    θ::BMPSWeights
end
BMPSPolicy(m::MetaMDP, θ) = BMPSPolicy(m, BMPSWeights(θ))

"Selects a computation to perform in a given belief."
(pol::BMPSPolicy)(b::Belief) = fast_act(pol, b)

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

function fast_act(pol::BMPSPolicy, b::Belief)
    θ = pol.θ
    voc = fast_voc(pol, b)
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
    # (3) 100000 samples of the VPI have been taken

    θ.vpi == 0. && return TERM  # no weight on VPI, VOC can't improve
    vpi = VPI(b)
    for i in 1:100000
        step!(vpi, 500)  # add 500 samples
        μ_voc = v + θ.vpi * vpi.μ
        σ_voc = θ.vpi * (vpi.σ / √vpi.n)
        (σ_voc < 1e-4 || abs(μ_voc - 0) > 3 * σ_voc) && break
        # (i == 100000) && println("Warning: VPI estimation did not converge.")
    end
    v + θ.vpi * vpi.μ > 0 && return c
    return TERM
end