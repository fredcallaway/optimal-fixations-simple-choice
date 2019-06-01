include("model.jl")
include("utils.jl")
using StatsBase
using StatsFuns

function voi1_sigma(lam, obs_lam)
    w = obs_lam / (lam + obs_lam)
    sample_sigma = √(1. / lam + 1. / obs_lam)
    w * sample_sigma
end

function slow_voi_n(b::Belief, c::Computation, n::Int)
    cv = competing_value(b.mu, c)
    obs_lam = b.obs_sigma ^ -2.
    d = Normal(b.mu[c], voi1_sigma(b.lam[c], obs_lam * n))
    expect_max_dist(d, cv) - maximum(b.mu)
end

function voi_n(b::Belief, c::Computation, n::Int)
    cv = competing_value(b.mu, c)
    obs_lam = b.obs_sigma ^ -2.
    µ, σ = b.mu[c], voi1_sigma(b.lam[c], obs_lam * n)
    α = (cv - µ) / σ
    p_improve = 1 - normcdf(α)
    p_improve < eps() && return 0.0
    amount_improve = µ + σ * normpdf(α) / p_improve
    new_ev = (1 - p_improve) * cv + p_improve * amount_improve
    new_ev - max(cv, µ)
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



# # %% ====================  ====================

# voc(pol, Belief(State(true_mdp)))
# pol = SoftBlinkered1(true_mdp, 10.)
# b = Belief(State(true_mdp))
# b.focused = 1
# println(voc(pol, b))
# countmap([pol(b) for i in 1:10000])
# using Glob
# include("job.jl")
# files = glob("runs/rando/jobs/*")
# # %% ====================  ====================
# display("")
# job = Job(files[15])
# m = MetaMDP(job)
# pol = optimized_policy(job)
# blinkered = Blinkered(m)
# Random.seed!(0)
# @time println(mean(rollout(pol).reward for i in 1:1000))
# Random.seed!(0)
# @time println(mean(rollout(blinkered).reward for i in 1:1000))
#
# # %% ====================  ====================
# b.mu[1] = 1
# b.focused = 2
# cost(m, b, c)
# voc_n(n) = voi_n(b, c, n) - (cost(m, b, c) + (n-1) * m.sample_cost)
# voc_n(1)
#
# # %% ====================  ====================
# y = [voi_n(b, 2, i) for i in x] .- m.sample_cost * x
# plot(x, y)
# hline!([voi1(b, 2) - m.sample_cost])
# # hline!([voi_action(b, 2)])
# hline!([voc_blinkered(m, b, 2)])
