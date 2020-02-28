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
        get_loss(pols, ds, x[end])
    end
end

top_xs = xs[rank[1:30]]
test_xs = repeat(top_xs, 30);
test_task = background("test"; save=true) do
    pmap(test_xs) do x
        loss(x)
    end
end
istaskdone(test_task)
test_results = fetch(test_task);

# %% ====================  ====================
test_results = deserialize("background_tasks/test")
reshape(combinedims(test_xs), 5, 30, 30)
L = reshape(combinedims(test_results), 2, 30, 30)
L = sum(L; dims=1) |> dropdims(1)
Lm = mean(L; dims=2) |> dropdims(2)
Ls = std(L; dims=2) |> dropdims(2)

top_rank = sortperm(Lm)
new_top = top_xs[top_rank]
new_top

mkpath("results/sobol/sims")
mkpath("results/sobol/sim_pols/")

sim_task = background("simulate") do
    pmap(enumerate(new_top)) do (i, x)
        both_sims = map(["two", "three"]) do num
            ds = build_dataset(num, -1)
            try
                pols = deserialize("results/sobol/sim_pols/$num$i")
            catch
                pols = get_policies(ds.n_item, x2prm(x))
                serialize("results/sobol/sim_pols/$num$i", pols)
            end
            map(pols) do pol
                simulate_test(pol, ds, x[end])
            end
        end
        serialize("results/sobol/sims/$i", both_sims)
        println("Wrote results/sobol/sims/$i")
    end
end




# %% ====================  ====================


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
