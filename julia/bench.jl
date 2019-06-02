# -*- coding: utf-8 -*-
# ---
# jupyter:
#   '@webio':
#     lastCommId: null
#     lastKernelId: null
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Julia 1.0.0
#     language: julia
#     name: julia-1.0
#   language_info:
#     file_extension: .jl
#     mimetype: application/julia
#     name: julia
#     version: 1.0.0
# ---

include("blinkered.jl")

using Profile
Profile.clear()

mdp = MetaMDP(obs_sigma=3, switch_cost=8, sample_cost=0.001)
pol = SoftBlinkered(mdp, 1000.)
b = Belief(State(mdp))

@time voc_blinkered(mdp, beliefs[1], c);

b = Belief([1., 1., 1.68], [2., 2., 4.], 3., 1)
c = 1
voc_n(n) = voi_n(b, c, n) - (cost(mdp, b, c) + (n-1) * mdp.sample_cost)
plot(voc_n.(1:100))

# +
beliefs = Belief[]
for i in 1:10000
    rollout(pol, callback=(b,c)->push!(beliefs, deepcopy(b)))
end
function f(pol, beliefs)
    for b in beliefs
        action_probs(pol, b)
    end
end

@time f(pol, beliefs)
# -

@profile f(pol, beliefs)

using ProfileView
ProfileView.view()

using HCubature
using Distributions
d = Normal(0, 1)
f(x) = pdf(d, x[1]) * max(0, x[1])

function expect_max_dist(d::Distribution, constant::Float64)
    p_improve = 1 - cdf(d, constant)
    p_improve < 1e-10 && return constant
    (1 - p_improve) * constant + p_improve * mean(Truncated(d, constant, Inf))
end

expect_max_dist(d, 0.)

cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
include("model.jl")


m = MetaMDP()
s = State(m)
b = Belief(s)
@time vpi(b);
vpi(b)


# %% ====================  ====================
using Glob
include("job.jl")
files = glob("runs/rando/jobs/*")
jobs = Job.(files)
policies = optimized_policy.(jobs) |> skipmissing |> collect
filter(policies) do pol
    pol.θ[end] > 0.5
end

-true
# %% ====================  ====================
using Cuba
# include("model.jl");
@time vpi(Belief(s), n_sample=50000);

# %% ====================  ====================
# g(x) =
# @time hcubature(g, zeros(3) .- 5, zeros(3) .+ 5, atol=1e-5);]
bel = Belief(s)
dists = Normal.(bel.mu, bel.lam .^ -0.5)
f(x) = maximum(x) * prod(pdf.(dists, x))
@time println(hcubature(f, ones(3) .* -4, ones(3) .* 4, atol=1e-5));


# %% ==================== Works 0.13 seconds ====================
pf(x) = maximum(x; dims=1) .* prod(pdf.(dists, x); dims=1)
# g(X)
# [f(X[:, 1]) f(X[:, 2])]
# f(x) = maximum(x; dims=1) .* prod(pdf.(dists, x); dims=1)
# X = randn(3, 8)
# x = zeros(8)
# x .= px(X)

function foo()
    a, b = -4., 4.
    mult = (b - a) ^ 3
    g(x, v) = begin
        x .= a .+ (b-a) .* x
        v .= pf(x) .* mult
    end
     # hcubature(g, zeros(3), ones(3), atol=1e-4);
     cuhre(g, 3, atol=1e-6, nvec=1000)
end

@time println(foo().integral)

# %% ==================== Works 0.22 seconds ====================
f(x) = maximum(x) * prod(pdf.(dists, x))
# f(x) = maximum(x; dims=1) .* prod(pdf.(dists, x); dims=1)

function foo()
    a, b = -4, 4
    mult = (b - a) ^ 3
    g(x) = begin
        # f(a .+ (b-a) .* x) * mult
        x .= a .+ (b-a) .* x
        y = f(x)
        y * mult
    end
     # hcubature(g, zeros(3), ones(3), atol=1e-4);
     cuhre((x, y) -> y[1] = g(x), 3, atol=1e-6)
end

@time println(foo().integral)

# %% ==================== Hmmm ====================
function new_vpi(b)
    μ, σ = b.mu, b.lam .^ -0.5
    dists = Normal.(μ, σ)
    low, high = μ .- (5 .* σ), μ .+ (5 .* σ)
    mult = prod(high - low)
    g(x, v) = begin
        x .= low .+ (high-low) .* x
        v .= maximum(x; dims=1) .* prod(pdf.(dists, x); dims=1) .* mult
    end
     # hcubature(g, zeros(3), ones(3), atol=1e-4);
     cuhre(g, 3, nvec=1000).integral[1] - maximum(μ)
end
b = Belief(s)
new_vpi(b)
# update!(b, 2, 0.2)
display("")
@time println(new_vpi(b));
println(vpi(b, n_sample=10000000));

update!(b, 1, 1)
println()
@time println(new_vpi(b));
println(vpi(b, n_sample=10000000));

using LinearAlgebra: det
# %% ====================  ====================
using HCubature
function new_vpi(b)
    μ, σ = b.mu, b.lam .^ -0.5
    dists = Normal.(μ, σ)
    low, high = μ .- (4 .* σ), μ .+ (4 .* σ)
    g(x) = begin
        maximum(x) * prod(pdf.(dists, x))
    end
     hcubature(g, low, high, atol=1e-5)[1] - maximum(μ)
     # cuhre(g, 3, atol=1e-5, nvec=1000).integral[1] - maximum(μ)
end
new_vpi(b)
@time println(new_vpi(b))
# %% ====================  ====================
X = rand(10000, 3)
X .-= 0.5
X .*= 2
mean(maximum(X; dims=2))
# %% ====================  ====================
b = Belief(s)
n_sample = 5000000
R = mem_zeros(n_sample, length(b.mu))
max_samples = mem_zeros(n_sample)

function foo()
    R[:] = randn(n_sample, length(b.mu))
    R .*= (b.lam .^ -0.5)' .+ b.mu'
    maximum!(max_samples, R)
    mean(max_samples) - maximum(b.mu)
end
println(foo())
@time println(foo())
