using StatsPlots
using SplitApplyCombine
using Serialization
include("utils.jl")
gr(label="")



# %% ====================  ====================
G = deserialize("G");
N = 5
# params = map(first, G.axes)
# @assert params == Symbol[:β_μ, :α, :σ_obs, :sample_cost, :switch_cost]
# params = reshape(split("prior noise cost switch"), 1, :)
params = reshape(split("alpha prior noise cost switch"), 1, :)

function best(X::Array, dims...)
    drop = [i for i in 1:N if i ∉ dims]
    B = minimum(X; dims=drop)
    permutedims(B, [dims...; drop]) |> dropdims((length(dims)+1:N)...)
end

X = G.L2 + G.L3;

best(X, 1)

thresh = minimum(X) + .02

map(1:5) do i
    best(X, i) .< thresh
end





# %% ====================  ====================
function plot_grid(X; ymax=nothing)
    mins, maxs = invert([juxt(minimum, maximum)(best(X, i, j)) for i in 1:N for j in i+1:N])
    lims = [minimum(mins), maximum(maxs)]
    if ymax != nothing
        lims[2] = ymax
    end

    P = map(1:N) do i
        map(1:N) do j
            if i == j
                plot(best(X, i), xlabel=params[i], ylims=lims)
            elseif i < j
                plot(axis=:off, grid=:off)
            else
                heatmap(best(X, i, j),
                    ylabel=params[i], xlabel=params[j],
                    colorbar=false, clim=Tuple(lims))
            end
        end
    end |> flatten

    plot(P..., size=(1000, 1000))
end


# %% ====================  ====================

plot_grid(G.L2)
savefig("figs/likelihood/new-grid-two")
plot_grid(G.L3)
savefig("figs/likelihood/new-grid-three")
plot_grid(G.L2 + G.L3)
savefig("figs/likelihood/new-grid-both")
plot_grid(G.L2 + G.L3; ymax=1.9)
savefig("figs/likelihood/new-grid-both-zoom")
