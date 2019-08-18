using Plots
include("meta_mdp.jl")
include("dc.jl")
using SplitApplyCombine
# %% ====================  ====================

function simulate(m::MetaMDP, v::Vector{Float64})
    policy = DirectedCognition(m)
    bs = Belief[]
    s = State(m, v)
    rollout(policy; state=s) do b, c
        push!(bs, copy(b))
    end
    bs
end

# %% ====================  ====================
m = MetaMDP(sample_cost=0.0005, switch_cost=0.02, σ_obs=3.)
v = [1,0,-1.]
# bs = simulate(m, v)

μ = combinedims([b.μ for b in bs])'
λ = combinedims([b.λ for b in bs])'
σ = λ .^ -0.5
focus = [b.focused for b in bs][2:end]
n = length(f)

colors = reshape(get_color_palette(:auto, plot_color(:white), 3), (1, 3))



fig1 = plot(μ, ribbon=σ,
    fillalpha=0.2, line=(2,),
    xlabel="Time",
    ylabel="Estimated value",
    lab="", color=colors, grid=:none,
    )

fig2 = plot(yticks=[], xticks=[], framestyle=:none, grid=:none)
for i in eachindex(focus)
    f = focus[i]
    plot!([i, i+1.1], [3,3], color=colors[f], line=(20,), lab="")
end

fig = plot(fig2, fig1, layout=grid(2,1, heights=[0.05,0.95]))
savefig("figs/illustrative/sampling.pdf")
fig
# %% ====================  ====================

y = 1:10
plot(y, ribbon=y/2, lab="estimate", fillalpha=0.1)
