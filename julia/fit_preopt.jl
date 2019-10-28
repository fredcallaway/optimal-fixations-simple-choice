include("results.jl")
include("pseudo_likelihood.jl")


@everywhere begin
    include("box.jl")
    μ_space = Dict(
        "fit" => (0, μ_emp),
        "emp" => μ_emp,
    )["emp"]

    fit_ε = Dict(
        "fit" => true,
        "one" => false,
    )["one"]

    index = Dict(
        "odd" => 1:2:length(trials),
        "even" => 2:2:length(trials),
        "all" => 1:length(trials),
    )["odd"]

    space = Box(
        :α => (50, 200, :log),
        :σ_obs => (1, 4),
        :sample_cost => (.001, .01, :log),
        :switch_cost => (.01, .05, :log),
        :µ => μ_emp,
        :σ => σ_emp,
        :sample_time => 100
        # (0, 2 * μ_emp),
        # :σ => (σ_emp / 4, 4 * σ_emp),
    )

    like_kws = (
        index = index,
        fit_ε = fit_ε,
        max_ε = 0.5
    )
end


# %% ==================== Load preoptimized policies ====================
using Glob

policies = map(glob("results/preopt_fine/policies/*")) do f
    try
        open(deserialize, f)
    catch
        missing
    end
end;

# xs = map(preopt) do pre
#     space(type2dict(pre.prm))
# end

# policies = [x.policy for x in preopt]
# serialize("tmp/preopt_policies", policies)
# serialize("tmp/preopt_x", xs)

# %% ==================== New metrics ====================

@everywhere metrics = [
    Metric(total_fix_time, 5),
    Metric(n_fix, Binning([0; 2:7; Inf])),
    Metric(t->t.choice, Binning(1:n_item+1)),
    Metric(t->propfix(t)[1], 5),
    Metric(t->propfix(t)[end], 5)
]

sims = map(rank_trials.value) do v
    sim_one(policy, prm, v)
end;

map(counts, invert(map(juxt(metrics...), rank_trials)))
map(counts, invert(map(juxt(metrics...), sims)))


# %% ====================  ====================

@everywhere function like(policy, σ_rating)
    logp, ε, baseline, lk = total_likelihood(policy,
        (μ=μ_emp, σ=σ_emp, sample_time=100, σ_rating=σ_rating);
        n_sim_hist=1000, parallel=false, metrics=metrics, like_kws...);
    logp / baseline
end


@time l0 = pmap(policies) do pol
    like(pol, 0)
end
println(policies[argmin(l0)])

@time l1 = pmap(policies) do pol
    like(pol, 1)
end

rank = sortperm(l0)
best = rank[1:50]

l0[old_best]

[best old_best]

x = [1, 2]

# %% ==================== Pre simulate ====================


@everywhere begin
    vs = unique(sort(t.value) for t in trials);
    sort!(vs, by=v->maximum(v) + std(v))  # fastest trials last for parallel efficiency

    function get_sims(policy, prm, vs; n=1000)
        map(vs) do v
            map(1:n) do i
                sim_one(policy, prm, v)
            end
        end
    end
end


@everywhere using DistributedArrays
@time all_sims = @DArray [get_sims(opt.policy, opt.prm, vs) for opt in preopt];


# %% ====================  ====================
@everywhere begin
    function likelihood_matrix(metrics, sims)
        apply_metrics = juxt(metrics...)
        histogram_size = Tuple(length(m.bins) for m in metrics)
        L = zeros(histogram_size...)
        for s in sims
            m = apply_metrics(s)
            L[m...] += 1
        end
        L ./ sum(L)
    end

    function make_likelihood(metrics, sims)
        Ls = [likelihood_matrix(metrics, sim) for sim in sims]
        Dict(zip(vs, Ls))
    end

    function total_likelihood(metrics, sims; fit_ε, index, max_ε, parallel=true, n_sim_hist=N_SIM_HIST)
        fit_trials = trials[index]

        likelihoods = make_likelihood(metrics, sims);
        apply_metrics = juxt(metrics...)
        histogram_size = Tuple(length(m.bins) for m in metrics)
        p_rand = 1 / prod(histogram_size)

        function likelihood(t::Trial)
            L = likelihoods[sort(t.value)]
            L[apply_metrics(t)...]
        end
        X = likelihood.(fit_trials);

        if fit_ε
            opt = Optim.optimize(0, max_ε) do ε
                -sum(@. log(ε * p_rand + (1 - ε) * X))
            end
            -opt.minimum, opt.minimizer
        else
            ε = prod(histogram_size) / (N_SIM_HIST + prod(histogram_size))
            sum(@. log(ε * p_rand + (1 - ε) * X)), ε
        end
    end
end

# %% ====================  ====================
# results = Results("fit_pseudo_preopt")

metrics = [
    Metric(rank_chosen, Binning(1:4)),
    Metric(total_fix_time, 10),
    # Metric(n_fix, 4),
    Metric(n_fix, Binning([0; 2:7; Inf])),
    # Metric(top_fix_proportion, 4)
]

like_kws = (fit_ε=true, index=eachindex(trials), max_ε=0.8)

# save(results, :metrics, metrics)
# save(results, :like_kws, like_kws)

@time likes, εs = map(all_sims) do sims
    total_likelihood(metrics, sims; like_kws...)
end |> collect |> invert



# %% ====================  ====================
histogram_size = Tuple(length(m.bins) for m in metrics)
p_rand = 1 / prod(histogram_size)
baseline = log(p_rand) * length(trials)

X = combinedims(xs)
y = likes ./ baseline

d = n_free(space)
test = 1:600
train = setdiff(eachindex(y), test);

model = GP(X[:, train], y[train], MeanConst(0.), Mat32Ard(zeros(d), 5.), -2.)
optimize!(model);

yhat, yvar = predict_f(model, X[:, test])
serialize("tmp/gp_cv", (X[:, test], y[test], yhat, yvar, model))


# %% ==================== Continue optimization ====================
function loss(x)
    prm = Params(space(x))
    @time policy = optimize_bmps(MetaMDP(prm); α=prm.α)
    @time total_likelihood(metrics, policy, prm;
        n_sim_hist=1000, like_kws...)[1] / baseline
end


opt_kws = (
    iterations=5,
    init_iters=0,
    acquisition="ei",
    optimize_every=5,
    acquisition_restarts=200,
    noisebounds=[-4, 1],
)
save(results, :opt_kws, opt_kws)

opt = gp_minimize(loss, n_free(space);
    init_Xy=(X,y), verbose=true, run=false, opt_kws...)
BayesianOptimization.optimizemodel!(opt.modeloptimizer, opt.model)

let
    mle = opt.model_optimizer |> space |> Params
    logp = opt.model_optimum
    @info "Iteration 0" mle logp
end

for i in 1:20
    boptimize!(opt)
    find_model_max!(opt)
    mle = opt.model_optimizer |> space |> Params
    logp = opt.model_optimum
    @info "Iteration $i" mle logp
end

find_model_max!(opt)
mle = opt.model_optimizer |> space |> Params
save(results, :mle, mle)
save(results, :opt, opt)


# %% ====================  ====================
prm = mle
@time policy = optimize_bmps(MetaMDP(prm); α=prm.α)
@time logp, ε, lk = total_likelihood(metrics, policy, prm; n_sim_hist=1000, like_kws...)

function empirical_likelihood()
    h = Dict(v => zeros(3) for v in keys(lk))
    for t in trials
        v = sort(t.value)
        r = metrics[1](t)
        h[v][r] += 1
    end
    for p in values(h)
        p ./= sum(p)
    end
    h
end

h = empirical_likelihood()


G = group(t->sort(t.value), trials)

# %% ====================  ====================

for t in G[[3., 7., 9.]]
    println(t.value, "  ", t.choice, "  ", t.value[t.choice], "  ", rank_chosen(t))
end


# %% ====================  ====================
function pretty(m::MetaMDP)
    println("Parameters")
    @printf "  σ_obs: %.2f\n  sample_cost: %.4f\n  switch_cost: %.4f\n" m.σ_obs m.sample_cost m.switch_cost
end
function pretty(pol::Policy)
    pretty(pol.m)
    @printf "  α: %.2f\n" pol.α
end
for i in rank[1:10]
    println()
    pretty(preopt[i].policy)
    println(ll[i])
end


