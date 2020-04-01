include("fit_base.jl")

jobs = sort(parse.(Int, readdir("$BASE_DIR/likelihood/")))
@assert jobs == collect(1:(GRID_SIZE^4))

outs = map(jobs) do i
    deserialize("$BASE_DIR/likelihood/$i")
end |> flatten;

@assert length(outs) == GRID_SIZE^5

function matrify(f)
    map(f, outs) |> reshape(repeat([GRID_SIZE], 5)...)
end

function get_dims()
    order = [:β_μ, :α, :σ_obs, :sample_cost, :switch_cost]
    P = matrify() do o
        o[1]
    end
    X = [
        P[:, 1, 1, 1, 1],
        P[1, :, 1, 1, 1],
        P[1, 1, :, 1, 1],
        P[1, 1, 1, :, 1],
        P[1, 1, 1, 1, :],
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
