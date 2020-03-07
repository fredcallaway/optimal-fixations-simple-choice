using Distributed

@everywhere begin
    using Glob
    using Serialization
    using SplitApplyCombine

    include("fit_base.jl")
    include("compute_policies.jl")
    include("compute_likelihood.jl")
end


# %% ==================== Load preopt results ====================

results = begin
    xs = map(glob("$BASE_DIR/likelihood/*")) do f
        endswith(f, "x") && return missing
        try
            deserialize(f)
        catch
            println("Can't read $f")
            missing
        end
    end |> skipmissing
    results = flatten(xs)
end;


prms, l2, l3, lc = map(results) do (prm, losses)
    prm, losses[1], losses[2], sum(losses)
end |> invert;


mkpath("$BASE_DIR/best_parameters/")
for (fit_mode, loss) in zip(["two", "three", "joint"], [l2, l3, lc])
    best = prms[partialsortperm(loss, 1:30)]
    serialize("$BASE_DIR/best_parameters/$fit_mode", best)
end



# %% ==================== Simulate ====================

@everywhere begin
    function compute_simulations(out::String, prms::Tuple{NamedTuple, NamedTuple})
        sims = map([2,3], prms) do n_item, prm
            trials = get_fold(load_dataset(n_item), LIKELIHOOD_PARAMS.test_fold, :test)
            policies = compute_policies(n_item, prm)
            prior = make_prior(trials, prm.β_μ)
            map(policies) do pol
                yield()
                simulate_trials(pol, prior, trials)
            end
        end
        serialize(out, sims)
        println("Wrote $out")
    end

    compute_simulations(out::String, prm::NamedTuple) = compute_simulations(out, (prm, prm))
end

task_both = background("sim_joint") do
    path = "$BASE_DIR/sim_joint"
    mkpath(path)
    top_joint = prms[partialsortperm(lc, 1:30)]
    pmap(1:30) do i
        compute_simulations("$path/$i", top_joint[i])
    end
end
# fetch(task_both)

task_sep = background("sim_sep") do
    path = "$BASE_DIR/sim_sep"
    mkpath(path)
    top_2 = prms[partialsortperm(l2, 1:30)]
    top_3 = prms[partialsortperm(l3, 1:30)]
    pmap(1:30) do i
        compute_simulations("$path/$i", (top_2[i], top_3[i]))
    end
end
# fetch(task_sep)
# %% ====================  ====================
tasks = map([:two, :three, :both]) do num
    l = Dict(:two => l2, :three => l3, :both => lc)[num]
    top = prms[partialsortperm(l, 1:30)]
    map(1:30) do i
        ("results/sim_$num")
end


# %% ==================== Summarize fits ====================
res = load_results("both")[1]
param_names = [free(load(res, :outer_space)); free(load(res, :inner_space))]

function load_x(res)
    args = load(res, :args)
    mle = load(res, :mle)
    os = load(res, :outer_space)
    is = load(res, :inner_space)
    t = type2dict(mle)
    vcat(os(t), is(t))
end

mkpath("figs/fits/")
function plot_fits(results, name; kws...)
    f = plot(
        xticks=(1:5, string.(param_names)),
        ylabel="Normalized parameter value",
        title=name)
    for res in results
        x = load_x(res)
        # plot!(x.x, color=lab[x.fold], lw=2)
        plot!(x, lw=2; ylim=(0,1), kws...)
    end
    savefig("figs/fits/$name.pdf")
    display(f)
end

open("old_fits.txt", "w") do f
    for dataset in ["two", "three", "both"]
        all_res = load_results(dataset, "OLD")
        println(f, "\n\n********* $dataset *********",)
        plot_fits(all_res, dataset)
        show(f, results_table(all_res), allcols=true, splitcols=false)
    end
end;

# %% ====================  ====================


pmap(enumerate()) do (i, x)
    both_sims = map(["two", "three"]) do num
        trials = load_dataset(num)
        pols = get_sim_pols(i, x, num)
        μ_emp, σ = empirical_prior(trials)
        β_μ = x[end]; μ = β_μ * μ_emp
        test_trials = trials[1:2:end]  # WARNING: ASSUMING PREDICT ODD
        map(pols) do pol
            simulate_trials(pol, test_trials, μ, σ)
        end
    end
    serialize("results/sobol/sims/$i", both_sims)
    println("Wrote results/sobol/sims/$i")
end

fetch(sim_task)

# sim_task = background("simulate") do
#     pmap(enumerate(new_top)) do (i, x)
#         both_sims = map(["two", "three"]) do num
#             trials = load_dataset(num)
#             pols = get_sim_pols(i, x, num)
#             μ_emp, σ = empirical_prior(trials)
#             β_μ = x[end]; μ = β_μ * μ_emp
#             test_trials = trials[1:2:end]  # WARNING: ASSUMING PREDICT ODD
#             map(pols) do pol
#                 simulate_trials(pol, test_trials, μ, σ)
#             end
#         end
#         serialize("results/sobol/sims/$i", both_sims)
#         println("Wrote results/sobol/sims/$i")
#     end
# end
# fetch(sim_task)



# %% ==================== Re-evaluate top 100 ====================

@everywhere function loss(n_item, prm)
    return 1.
    policies = compute_policies(n_item, prm)
    logp, ε, baseline = likelihood(policies, prm.β_μ; LIKELIHOOD_PARAMS..., n_sim_hist=100_000, fold=:train)
    logp / baseline
end

@everywhere function loss(prm)
    mapreduce(+, [2, 3]) do n_item
        loss(n_item, prm)
    end
end

rank = sortperm(lc);
top_prms = prms[rank[1:100]]
test_prms = repeat(top_prms, 10);
@time test_results = pmap(enumerate(test_prms)) do (i, prm)
    println("Begin $i")
    loss(prm)
end

# %% ====================  ====================
test_results = deserialize("background_tasks/test")
L = reshape(combinedims(test_results), 2, 30, 30)
L = sum(L; dims=1) |> dropdims(1)
Lm = mean(L; dims=2) |> dropdims(2)
Ls = std(L; dims=2) |> dropdims(2)

top_rank = sortperm(Lm)
new_top = top_xs[top_rank]






# %% ====================  ====================
run_name = "all_sobol"
combined_sims = map(1:3) do i
    map(deserialize("results/sobol/sims/$i")) do sims
        reduce(vcat, sims)
    end
end |> invert;

reduce(vcat, )

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
