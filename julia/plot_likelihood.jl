using StatsPlots
using SplitApplyCombine
using Serialization
include("utils.jl")
gr(label="")



# %% ====================  ====================
G = deserialize("G");

params = map(first, G.axes)
@assert params == Symbol[:β_μ, :σ_obs, :sample_cost, :switch_cost]
params = reshape(split("prior noise cost switch"), 1, :)

function best(X::Array, dims...)
    drop = [i for i in 1:4 if i ∉ dims]
    B = minimum(X; dims=drop)
    permutedims(B, [dims...; drop]) |> dropdims((length(dims)+1:4)...)
end


# %% ====================  ====================
function plot_grid(X; ymax=nothing)
    mins, maxs = invert([juxt(minimum, maximum)(best(X, i, j)) for i in 1:4 for j in i+1:4])
    lims = [minimum(mins), maximum(maxs)]
    if ymax != nothing
        lims[2] = ymax
    end

    P = map(1:4) do i
        map(1:4) do j
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
savefig("grid-two")
plot_grid(G.L3)
savefig("grid-three")
plot_grid(G.L2 + G.L3)
savefig("grid-both")
plot_grid(G.L2 + G.L3; ymax=1.9)
savefig("grid-both-zoom")
