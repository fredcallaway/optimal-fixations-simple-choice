using Glob
using Serialization
using SplitApplyCombine
using GaussianProcesses

@everywhere include("preopt_core.jl")




# %% ==================== Load preopt results ====================

results = begin
    xs = map(glob("results/like_ucb/*")) do f
        endswith(f, "x") && return missing
        try
            deserialize(f)
        catch
            println("Can't read $f")
            missing
        end
    end |> skipmissing
    xs = filter(x-> x != "started", xs)
    results = flatten(xs)
end;

xs, y = map(results) do (prm, losses)
    x = [space(type2dict(prm)); prm.β_μ]
    x, sum(losses)
end |> invert
X = combinedims(xs)
rank = sortperm(y)

# %% ====================  ====================
ds = build_dataset("two", -1);

pols = get_policies(ds.n_item, x2prm(x))
losses = get_loss(pols, ds, x[end])

@everywhere function loss(x)
    datasets = [build_dataset("two", -1), build_dataset("three", -1)];
    map(datasets) do ds
        pols = get_policies(ds.n_item, x2prm(x))
        get_loss(pols, ds)
    end
end

test_xs = repeat(xs[rank[1:30]], 30);
test_task = background("test"; save=true) do
    pmap(test_xs) do x
        loss(x)
    end
end
# test_results = fetch(test_task);


# test_xs2 = repeat(xs[rank[1:30]], 100);
# test_task2 = background("test"; save=true) do
#     pmap(test_xs2) do x
#         loss(x)
#     end
# end
# test_results2 = fetch(test_task2);

serialize("tmp/Xy", (X, y))
# %% ====================  ====================

rank = sortperm(y)
y[rank[1:30]]
X[:, rank[1:30]]


job = div(rank[1], 10)
ucb_out = deserialize("results/preopt_ucb/$job");
ucb_out[1][1][1]



datasets = [build_dataset("two", -1), build_dataset("three", -1)]
map(1:2) do item_idx
    policies, μ, sem = ucb[item_idx]
    ds = datasets[item_idx]
    top = policies[partialsortperm(-μ, 1:n_top)]
    prm = (β_μ=β_μ, β_σ=1., σ_rating=NaN)
    logp, ε, baseline = likelihood(ds, top, prm; parallel=false);
    logp / baseline
end



# %% ====================  ====================



serialize("tmp/top10", (
    x = xs[partialsortperm(y, 1:10)],
    params = [free(space); :β_μ]
))


gp = GPE(X, y, Mat32Ard(zeros(size(X, 1)), 5.))
