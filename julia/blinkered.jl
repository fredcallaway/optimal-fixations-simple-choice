include("model.jl")

m = MetaMDP(switch_cost=3)
s = State(m)
b = Belief(s)

function voi1_sigma(lam, obs_sigma)
    obs_lam = obs_sigma ^ -2
    w = obs_lam / (lam + obs_lam)
    sample_sigma = √(1/lam + 1/obs_lam)
    w * sample_sigma
end

function voi1(b::Belief, c::Computation)
    cv = competing_value(b.mu, c)
    d = Normal(b.mu[c], voi1_sigma(b.lam[c], b.obs_sigma[c]))
    expect_max_dist(d, cv) - maximum(b.mu)
end

voi1_sigma(1, m.obs_sigma/√1)

function voi_n(b::Belief, c::Computation, n::Int)
    cv = competing_value(b.mu, c)
    d = Normal(b.mu[c], voi1_sigma(b.lam[c], b.obs_sigma[c] / √n))
    expect_max_dist(d, cv) - maximum(b.mu)
end

function int_line_search(start, f)
    i = start; last_val = -Inf; val = f(start)
    while val > last_val
        i += 1
        last_val, val = val, f(i)
    end
    (i-1, last_val)
end

function voc_blinkered(m::MetaMDP, b::Belief, c::Computation)
    voc_n(n) = voi_n(b, c, n) - (cost(m, b, c) + (n-1) * m.sample_cost)
    int_line_search(1, voc_n)[2]
end

struct Blinkered
    m::MetaMDP
end
(π::Blinkered)(b::Belief) = begin
    voc = [voc_blinkered(π.m, b, c) for c in 1:π.m.n_arm]
    v, c = findmax(noisy(voc))
    v <= 0 ? TERM : c
end

# # %% ====================  ====================
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
