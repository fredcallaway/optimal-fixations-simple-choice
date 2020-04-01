using StatsPlots
using SplitApplyCombine
using Serialization
include("utils.jl")
gr(label="")



# %% ====================  ====================
G = deserialize("results/grid17/grid");
raw_params = map(first, G.dims)
@assert raw_params == Symbol[:β_μ, :α, :σ_obs, :sample_cost, :switch_cost]
ndim = length(raw_params)
params = reshape(split("prior alpha noise cost switch"), 1, :)
# %% ====================  ====================
function best(X::Array, dims...; ymax=Inf)
    drop = [i for i in 1:ndim if i ∉ dims]
    B = minimum(X; dims=drop)
    b = permutedims(B, [dims...; drop]) |> dropdims((length(dims)+1:ndim)...)
    b[b .> ymax] .= NaN
    b
end

function get_ticks(i)
    idx = 1:7
    vals = round.(G.dims[i][2]; sigdigits=2)
    idx[1:2:end], vals[1:2:end]
end

function plot_grid(X; ymax=Inf)
    mins, maxs = invert([juxt(minimum, maximum)(best(X, i, j)) for i in 1:ndim for j in i+1:ndim])
    lims = [minimum(mins), maximum(maxs)]

    if ymax != Inf
        lims[2] = ymax
    end

    P = map(1:ndim) do i
        map(1:ndim) do j
            if i == j
                plot(best(X, i), xlabel=params[i], ylims=lims, xticks=get_ticks(i))
            elseif i < j
                plot(axis=:off, grid=:off)
            else
                heatmap(best(X, i, j; ymax=ymax),
                    xlabel=params[j],
                    ylabel=params[i], 
                    xticks=get_ticks(j),
                    yticks=get_ticks(i),
                    colorbar=false, clim=Tuple(lims))
            end
        end
    end |> flatten

    plot(P..., size=(1000, 1000))
end

plot_grid(G.L2)

# %% ====================  ====================
out = "/Users/fred/papers/attention-optimal-sampling/figs"
mkpath(out)
savefig("$out/grid-two.pdf")
plot_grid(G.L3)
savefig("$out/grid-three.pdf")
plot_grid(G.L2 + G.L3)
savefig("$out/grid-both.pdf")
