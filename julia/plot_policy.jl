using Distributed
using StatsPlots
using LaTeXStrings
using Serialization
using Plots.Measures
nprocs() == 1 && addprocs()
pyplot(label="")
@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("binning.jl")
end
policies = deserialize("results/sobol4/simulation_policies/joint-true/1");
pol2 = policies[1][1]
pol3 = policies[2][1]

# %% ====================  ====================
Plots.scalefontsizes()
Plots.scalefontsizes(3)
function ticklabels(x)
    q = Int.(quantile(1:length(x), [0,.5,1]))
    # q = Int.(quantile(1:length(x), [0,1]))
    eachindex(x)[q], Int.(x[q])
end
plot([1,2])

# %% ==================== Artifically setting μ and λ ====================
@everywhere function sample_bs(pol, n_roll)
    bs = Belief[]
    for i in 1:n_roll
        rollout(pol, callback=(b,c)->push!(bs, deepcopy(b)))
    end
    bs
end

@everywhere function modify(μ, x)
    if length(μ) == 2
        μ = copy(μ)
        μ[1] = μ[2] + 2x
        return μ
    end
    μ = sort(μ)
    if x < 0
        μ[1] = μ[2] + x
    elseif x > 0
        μ[1], μ[3] = μ[3], μ[1]
        μ[1] = μ[2] + x
    else
        μ[1], μ[2] = μ[2], μ[1]
    end
    μ
end

for b in vcat(sample_bs(pol3, 100), sample_bs(pol2, 100))
    x = randn()
    μ = modify(b.μ, x)
    try
        @assert μ[1] - median(μ) ≈ x
    catch e
        error("failure $x $(b.μ) $μ")
    end
end
# %% ====================  ====================
μs = -1:0.005:1
λs = 1:0.01:5
# %% ====================  ====================

function make_X(pol, n_roll=500)
    bs = map(sample_bs(pol, n_roll)) do b
        b.focused = -1
        b
    end
    pmap(Iterators.product(μs, λs)) do (x, λ)
        length(bs) \ mapreduce(+, bs) do b
            b = deepcopy(b)
            b.μ = modify(b.μ, x)
            b.λ[1] = λ
            v = [voc(pol, b); 0.]
            softmax(1e4 .* v)[1]
        end
    end |> transpose |> collect
end

using NPZ
mkpath("results/hard_policy_npy")
npzwrite("results/hard_policy_npy/2", make_X(pol2))
npzwrite("results/hard_policy_npy/3", make_X(pol3))
npzwrite("results/hard_policy_npy/μs", collect(μs))
npzwrite("results/hard_policy_npy/λs", collect(λs))

# %% ====================  ====================
h1 = heatmap(X, xticks=ticklabels(μs), yticks=ticklabels(λs),
    # xlabel=L"\mu^{(1)} - \max(\mu^{(2)}, \mu^{(3)})",
    # ylabel=L"λ^{(1)}",
    xlabel="Estimated value difference\n" *
     # L"\mu^{(1)} - \mu^{(2)}",
     L"\mu^{(1)} - \mathrm{median}(\mu)",
    ylabel="Estimate certainty  " * L"λ^{(1)}",
    colorbar_title="P(sample item 1)",
    clims=clims,
    size=(900,700),
    aspect_ratio=1,
    # right=50mm,
    # margin=100mm,
    title="Two items",
    )


# %% ====================  ====================
@time X2 = make_X(pol2)
# @time X3 = make_X(pol3)

# %% ====================  ====================

using Plots: px
clims = (0, maximum(X2))
h1 = heatmap(X2, xticks=ticklabels(μs), yticks=ticklabels(λs),
    # xlabel=L"\mu^{(1)} - \max(\mu^{(2)}, \mu^{(3)})",
    # ylabel=L"λ^{(1)}",
    xlabel="Estimated value difference\n" *
     # L"\mu^{(1)} - \mu^{(2)}",
     L"\mu^{(1)} - \mathrm{median}(\mu)",
    ylabel="Estimate certainty  " * L"λ^{(1)}",
    colorbar_title="P(sample item 1)",
    clims=clims,
    size=(900,700),
    aspect_ratio=1,
    # right=50px,
    title="Two items",
    )
savefig("figs/policy2.png")

# %% ====================  ====================
h2 = heatmap(X3, xticks=ticklabels(μs), yticks=ticklabels(λs),
    # xlabel=L"\mu^{(1)} - \max(\mu^{(2)}, \mu^{(3)})",
    # ylabel=L"λ^{(1)}",
    xlabel="Estimated value difference\n" *
    # L"\mu^{(1)} - \max(\mu^{(2)}, \mu^{(3)})",
    L"\mu^{(1)} - \mathrm{median}(\mu)",
    ylabel="Estimate certainty  " * L"λ^{(1)}",
    colorbar_title="P(sample item 1)",
    clims=clims,
    size=(900,700),
    aspect_ratio=1,
    title=("Three items")
    )
savefig("figs/policy3.png")

# mkpath("results/plot_policy")
# %% ====================  ====================


function term_reward_curve(N)
    m = MetaMDP(sample_cost=0., σ_obs=2)
    pol = BMPSPolicy(m, [0., 1., 0., 0.])
    trc = @distributed (+) for i in 1:Int(N/100)
        x = zeros(101)
        for i in 1:100
            roll = rollout(pol; max_steps=101) do b, c
                i = 1+Int((sum(b.λ) - 3.) / (m.σ_obs^-2))
                x[i] += term_reward(b)
            end
            @assert roll.steps == 101
        end
        x
    end
    trc ./ N
end

function line!(v, lab, c=:black)
    hline!([v], c=c, line=(:dash, 3))
    annotate!(50, v, text(lab, 30, :center, :bottom), c=c)
end

# %% ====================  ====================
trc = term_reward_curve(Int(1e6))
# %% ====================  ====================
plot(0:100, trc, c=:black, ylim=(0, 1),size=(900,700),
lw=2, xlabel="Number of Computations", ylabel="Value of Chosen Item")
line!(vpi(Belief(m)), L"\mathrm{VOI}_\mathrm{full}")
line!(voi1(Belief(m), 1), L"\mathrm{VOI}_\mathrm{myopic}")
# line!(voi_action(Belief(m), 1), L"\mathrm{VOI}_\mathrm{item}")
savefig("figs/voi-vpi.pdf")
savefig("figs/voi-vpi.png")
# %% ====================  ====================
pol = pol2
bs = map(sample_bs(pol, 500)) do b
    b.focused = -1
    b
end

# %% ====================  ====================
b = Belief(pol2.m)

function foo(x, λ)
    length(bs) \ mapreduce(+, bs) do b
        b = deepcopy(b)
        b.μ = modify(b.μ, x)
        b.λ[1] = λ
        # b.λ[2] = rand(λs)
        # b.μ = [0, 0]
        # cv = maximum(b.μ[2:end])
        # b.μ[1] = μ + cv; b.λ[1] = λ
        v = [voc(pol, b); 0.]
        softmax(pol.α .* v)[1]
    end
end


function dofoo(f, xs, λ)
    map(xs) do x
        length(bs) \ mapreduce(+, bs) do b
            b = deepcopy(b)
            b.μ = modify(b.μ, x)
            b.λ[1] = λ
            f(b)
        end
    end
end


x = range(-0.2, 0.2, length=30)
y = dofoo(x, 4.) do b
    voi1(b, 1)
end
plot(x, y)
y = dofoo(x, 4.) do b
    voi_action(b, 1)
end
plot!(x, y)
y = dofoo(x, 4.) do b
    vpi(b)
end
plot!(x, y)
