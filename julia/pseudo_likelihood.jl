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
    const the_metrics = [
        Metric(total_fix_time, 10),
        Metric(n_fix, Binning([0; 2:7; Inf])),
        Metric(rank_chosen, Binning(1:4)),
        # Metric(top_fix_proportion, 10)
    ]
    const N_SIM_HIST = 10_000

    const max_steps = Int(cld(maximum(total_fix_time.(trials)), 100))


    function sim_one(policy, prm, v)
        sim = simulate(policy, (v .- prm.μ) ./ prm.σ; max_steps=max_steps)
        fixs, fix_times = parse_fixations(sim.samples, prm.sample_time)
        (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times)
    end

    function get_metrics(metrics, policy, prm, v, N, parallel)
        apply_metrics = juxt(metrics...)

        if parallel
            ms = @distributed vcat for i in 1:N
                sim = sim_one(policy, prm, v)
                apply_metrics(sim)
                # n_fix(sim) == 0 ? missing :
            end
        else
            ms = map(1:N) do i
                sim = sim_one(policy, prm, v)
                apply_metrics(sim)
                # n_fix(sim) == 0 ? missing :
            end
        end
        skipmissing(ms)
    end

    function likelihood_matrix(metrics, policy, prm, v::Vector{Float64}; N=N_SIM_HIST, parallel=true)
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

    # const P_RAND = 1 / prod(histogram_size)
    # const BASELINE = log(P_RAND) * length(trials)

    function total_likelihood(policy, prm; fit_ε, index, max_ε, metrics=the_metrics, parallel=true, n_sim_hist=N_SIM_HIST)
        fit_trials = trials[index]
        vs = unique(sort(t.value) for t in fit_trials);
        sort!(vs, by=std)  # fastest trials last for parallel efficiency
        out = (parallel ? asyncmap : map)(vs) do v
            likelihood_matrix(metrics, policy, prm, v; parallel=parallel, N=n_sim_hist)
        end
        likelihoods = Dict(zip(vs, out))

        apply_metrics = juxt(metrics...)
        histogram_size = Tuple(length(m.bins) for m in metrics)
        p_rand = 1 / prod(histogram_size)
        baseline = log(p_rand) * length(index)

        function likelihood(policy, t::Trial)
            L = likelihoods[sort(t.value)]
            L[apply_metrics(t)...]
        end

        X = likelihood.([policy], fit_trials);

        f(ε) = sum(@. log(ε * p_rand + (1 - ε) * X))

        if fit_ε
            ε = Optim.optimize(ε->-f(ε), 0, max_ε).minimizer
        else
            ε = prod(histogram_size) / (N_SIM_HIST + prod(histogram_size))
        end
        f(ε), ε, baseline, likelihoods
    end
end
