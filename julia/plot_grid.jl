using StatsPlots
using Plots.Measures
using SplitApplyCombine
using Serialization
include("utils.jl")
# gr(label="")

using LaTeXStrings

# %% ====================  ====================
G = deserialize("results/grid13/grid");
raw_params = map(first, G.dims)
@assert raw_params == Symbol[:α, :σ_obs, :sample_cost, :switch_cost]
ndim = length(raw_params)
# params = reshape(split("alpha noise cost switch"), 1, :)
params =  [L"\beta", L"\sigma_x", L"\gamma_\mathrm{sample}", L"\gamma_\mathrm{switch}"]
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
    idx[1:3:end], vals[1:3:end]
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
                    colorbar=false, clim=Tuple(lims),  aspect_ratio = 1)
            end
        end
    end |> flatten

    plot(P..., size=(1100, 1000), right_margin=4mm)
end

# %% ====================  ====================
# out = "/Users/fred/papers/attention-optimal-sampling/figs"
out = "figs/grid"
mkpath(out)
plot_grid(G.L2)
savefig("$out/two.png")
plot_grid(G.L3)
savefig("$out/three.png")
plot_grid(G.L2 + G.L3)
savefig("$out/both.png")
