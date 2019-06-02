include("model.jl")
include("utils.jl")
using StatsBase
# m = MetaMDP(switch_cost=3)
# s = State(m)
# b = Belief(s)

function _voi1_sigma(lam, obs_sigma)
    obs_lam = obs_sigma ^ -2
    w = obs_lam / (lam + obs_lam)
    sample_sigma = √(1/lam + 1/obs_lam)
    w * sample_sigma
end

function _voi_n(b::Belief, c::Computation, n::Int)
    cv = competing_value(b.mu, c)
    d = Normal(b.mu[c], _voi1_sigma(b.lam[c], b.obs_sigma / √n))
    expect_max_dist(d, cv) - maximum(b.mu)
end

function _voc_blinkered(m::MetaMDP, b::Belief, c::Computation)
    c == TERM && return 0.
    voc_n(n) = _voi_n(b, c, n) - (cost(m, b, c) + (n-1) * m.sample_cost)
    # int_line_search(1, voc_n)[2]
    maximum(voc_n.(1:100))
end

struct Blinkered
    m::MetaMDP
end
(pol::Blinkered)(b::Belief) = begin
    voc = [_voc_blinkered(pol.m, b, c) for c in 1:pol.m.n_arm]
    v, c = findmax(noisy(voc))
    v <= 0 ? TERM : c
end

# %% ====================  ====================
struct SoftBlinkered
    m::MetaMDP
    α::Float64
end
voc(pol::SoftBlinkered, b::Belief) = [_voc_blinkered(pol.m, b, c) for c in 0:pol.m.n_arm]
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
# voc_n(n) = _voi_n(b, c, n) - (cost(m, b, c) + (n-1) * m.sample_cost)
# voc_n(1)
#
# # %% ====================  ====================
# y = [_voi_n(b, 2, i) for i in x] .- m.sample_cost * x
# plot(x, y)
# hline!([voi1(b, 2) - m.sample_cost])
# # hline!([voi_action(b, 2)])
# hline!([_voc_blinkered(m, b, 2)])
