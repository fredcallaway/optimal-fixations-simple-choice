include("model.jl")
include("utils.jl")
using StatsBase

function voi1_sigma(lam, obs_sigma)
    obs_lam = obs_sigma ^ -2
    w = obs_lam / (lam + obs_lam)
    sample_sigma = √(1/lam + 1/obs_lam)
    w * sample_sigma
end

function voi_n(b::Belief, c::Computation, n::Int)
    cv = competing_value(b.mu, c)
    d = Normal(b.mu[c], voi1_sigma(b.lam[c], b.obs_sigma / √n))
    expect_max_dist(d, cv) - maximum(b.mu)
end


function voc_blinkered(m::MetaMDP, b::Belief, c::Computation)
    c == TERM && return 0.
    voc_n(n) = voi_n(b, c, n) - (cost(m, b, c) + (n-1) * m.sample_cost)
    # int_line_search(1, voc_n)[2]
    maximum(voc_n.(1:100))
end

struct Blinkered
    m::MetaMDP
end
(pol::Blinkered)(b::Belief) = begin
    voc = [voc_blinkered(pol.m, b, c) for c in 1:pol.m.n_arm]
    v, c = findmax(noisy(voc))
    v <= 0 ? TERM : c
end

# %% ====================  ====================
struct SoftBlinkered
    m::MetaMDP
    α::Float64
end
voc(pol::SoftBlinkered, b::Belief) = [voc_blinkered(pol.m, b, c) for c in 0:pol.m.n_arm]
action_probs(pol::SoftBlinkered, b::Belief) = softmax(pol.α * voc(pol, b))

(pol::SoftBlinkered)(b::Belief) = begin
    sample(0:pol.m.n_arm, Weights(action_probs(pol, b)))
end
