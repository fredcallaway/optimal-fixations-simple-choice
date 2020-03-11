

outs = map(1:10000) do i
    try
        xs = deserialize("results/big-grid/likelihood/$i")
        @assert xs[end][1].β_μ < 1
        xs
    catch
        x = ((α = NaN, σ_obs = NaN, sample_cost = NaN, switch_cost = NaN, β_μ = NaN), [1., 1.])
        [x for i in 1:10]
    end
end |> flatten;

n_bad = mapreduce(+, outs) do o
    isnan(o[1].α)
end

# %% ====================  ====================
function matrify(f)
    map(f, outs) |> reshape(10, 10, 10, 10, 10)
end

P = matrify() do o
    o[1]
end;

order = [:β_μ, :α, :σ_obs, :sample_cost, :switch_cost]
dims = [
    P[:, 1, 1, 1, 1],
    P[1, :, 1, 1, 1],
    P[1, 1, :, 1, 1],
    P[1, 1, 1, :, 1],
    P[1, 1, 1, 1, :],
];

map(order, dims) do s, d
    s => getfield.(d, s)
end



L2 = matrify() do o
    o[2][1]
end;
L3 = matrify() do o
    o[2][2]
end;

space = Box(
    :sample_time => 100,
    # :α => (50, 200),
    :α => 200,
    :σ_obs => (2, 4),
    :sample_cost => (.002, .006),
    :switch_cost => (.013, .025),
    :β_μ => (0,1)
)




order = [:β_μ, :σ_obs, :sample_cost, :switch_cost]
g = 0:0.1:1




serialize("results/grid/G", (
    # axes=[k => inscale.(g, space[k]...) for k in order],
    L2 = L2,
    L3 = L3
))