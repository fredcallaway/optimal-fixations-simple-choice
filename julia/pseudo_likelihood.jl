using Distributed

@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("human.jl")
    include("binning.jl")
    include("simulations.jl")
    include("features.jl")
    include("params.jl")
    include("metrics.jl")
end


@everywhere begin
    const N_SIM_HIST = 10_000
    const MAX_STEPS = 200  # 20 seconds

    function sim_one(policy, prm, v)
        sim = simulate(policy, (v .- prm.μ) ./ prm.σ; max_steps=MAX_STEPS)
        fixs, fix_times = parse_fixations(sim.samples, prm.sample_time)
        (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times)
    end

    function get_metrics(metrics, policy, prm, v, N, parallel)
        apply_metrics = juxt(metrics...)

        if parallel
            ms = @distributed vcat for i in 1:N
                sim = sim_one(policy, prm, v .+ prm.σ_rating .* randn(length(v)))
                apply_metrics(sim)
                # n_fix(sim) == 0 ? missing :
            end
        else
            ms = map(1:N) do i
                sim = sim_one(policy, prm, v .+ prm.σ_rating .* randn(length(v)))
                apply_metrics(sim)
                # n_fix(sim) == 0 ? missing :
            end
        end
        skipmissing(ms)
    end

    function likelihood_matrix(metrics, policy, prm, v; N=N_SIM_HIST, parallel=true)
        histogram_size = Tuple(length(m.bins) for m in metrics)
        apply_metrics = juxt(metrics...)
        L = zeros(histogram_size...)
        for m in get_metrics(metrics, policy, prm, v, N, parallel)
            L[m...] += 1
        end
        L ./ sum(L)
    end
end

# %% ====================  ====================
@everywhere begin
    using Optim

    function total_likelihood(policy, prm, trials; fit_ε, max_ε, metrics, parallel=true, n_sim_hist=N_SIM_HIST)
        vs = unique(sort(t.value) for t in trials);
        sort!(vs, by=std)  # fastest rank_trials last for parallel efficiency
        out = (parallel ? asyncmap : map)(vs) do v
            likelihood_matrix(metrics, policy, prm, v; parallel=parallel, N=n_sim_hist)
        end
        likelihoods = Dict(zip(vs, out))

        apply_metrics = juxt(metrics...)
        histogram_size = Tuple(length(m.bins) for m in metrics)
        p_rand = 1 / prod(histogram_size)
        baseline = log(p_rand) * length(trials)

        function likelihood(policy, t)
            L = likelihoods[sort(t.value)]
            L[apply_metrics(t)...]
        end

        X = likelihood.([policy], trials);

        f(ε) = sum(@. log(ε * p_rand + (1 - ε) * X))

        if fit_ε
            ε = Optim.optimize(ε->-f(ε), 0, max_ε).minimizer
        else
            ε = prod(histogram_size) / (n_sim_hist + prod(histogram_size))
        end
        f(ε), ε, baseline, likelihoods
    end
end
