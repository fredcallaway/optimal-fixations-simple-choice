include("fit_base.jl")
using SplitApplyCombine

# jobs = sort(parse.(Int, readdir("$BASE_DIR/likelihood/")))
# @assert jobs == collect(1:(GRID_SIZE^4))
jobs = 1:2401
SKIPMISSING = true
outs = map(jobs) do i
    if SKIPMISSING && !isfile("$BASE_DIR/likelihood/$i")
        return [((α = NaN, σ_obs =NaN, sample_cost =NaN, switch_cost =NaN, β_μ =NaN), [50000., 50000.])]
    end
    deserialize("$BASE_DIR/likelihood/$i")
end |> flatten;

function matrify(f)
    map(f, outs) |> reshape(repeat([GRID_SIZE], 4)...)
end

function get_dims()
    order = [:α, :σ_obs, :sample_cost, :switch_cost]
    P = matrify() do o
        o[1]
    end
    X = [
        P[:, 1, 1, 1],
        P[1, :, 1, 1],
        P[1, 1, :, 1],
        P[1, 1, 1, :],
    ];

    map(order, X) do s, d
        s => getfield.(d, s)
    end

end

L2 = matrify() do o
    o[2][1]
end

L3 = matrify() do o
    o[2][2]
end

serialize("$BASE_DIR/grid", (
    dims=get_dims(),
    L2 = L2,
    L3 = L3
))
