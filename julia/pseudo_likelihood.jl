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
end


@everywhere begin
    const N_SIM_HIST = 10_000

    total_fix_time(t) = sum(t.fix_times)
    rank_chosen(t) = sortperm(t.value; rev=true)[t.choice]
    n_fix(t) = length(t.fixations)
    function chosen_fix_proportion(t)
        tft = total_fix_times(t)
        tft ./= sum(tft)
        tft[t.choice]
    end

    function top_fix_proportion(t)
        tft = total_fix_times(t)
        tft ./= sum(tft)
        tft[argmax(t.value)]
    end

    struct Metric{F}
        f::F
        bins::Binning
    end

    function Metric(f::Function, n::Int)
        bins = Binning(f.(trials), n)
        bins.limits[1] = -Inf; bins.limits[end] = Inf
        Metric(f, bins)
    end
    (m::Metric)(t) = t |> m.f |> m.bins

    # m = Metric(total_fix_time, 10)
    # counts(m.(trials))  # something's wrong?

    const the_metrics = [
        Metric(total_fix_time, 10),
        Metric(n_fix, Binning([1:7; Inf])),
        Metric(rank_chosen, Binning(1:4)),
        # Metric(top_fix_proportion, 10)
    ]

    apply_metrics = juxt(the_metrics...)

    const max_steps = Int(cld(maximum(total_fix_time.(trials)), 100))
    const histogram_size = Tuple(length(m.bins) for m in the_metrics)

    function sim_one(policy, prm, v)
        sim = simulate(policy, (v .- prm.μ) ./ prm.σ; max_steps=max_steps)
        fixs, fix_times = parse_fixations(sim.samples, prm.sample_time)
        (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times)
    end

    function get_metrics(policy, prm, v, N, parallel)
        if parallel
            ms = @distributed vcat for i in 1:N
                sim = sim_one(policy, prm, v)
                n_fix(sim) == 0 ? missing : apply_metrics(sim)
            end
        else
            ms = map(1:N) do i
                sim = sim_one(policy, prm, v)
                n_fix(sim) == 0 ? missing : apply_metrics(sim)
            end
        end
        skipmissing(ms)
    end
    function likelihood_matrix(policy, prm, v::Vector{Float64}; N=N_SIM_HIST, parallel=true)
        L = zeros(histogram_size...)
        for m in get_metrics(policy, prm, v, N, parallel)
            L[m...] += 1
        end
        L ./ sum(L)
    end
end

# %% ====================  ====================
@everywhere begin
    using Optim

    const P_RAND = 1 / prod(histogram_size)
    const BASELINE = log(P_RAND) * length(trials)

    function total_likelihood(policy, prm; fit_ε, index, parallel=true)
        fit_trials = trials[index]
        vs = unique(sort(t.value) for t in fit_trials);
        sort!(vs, by=std)  # fastest trials last for parallel efficiency
        out = (parallel ? asyncmap : map)(vs) do v
            likelihood_matrix(policy, prm, v; parallel=parallel)
        end

        likelihoods = Dict(zip(vs, out))
        function likelihood(policy, t::Trial)
            L = likelihoods[sort(t.value)]
            L[apply_metrics(t)...]
        end

        X = likelihood.([policy], fit_trials);

        if fit_ε
            opt = Optim.optimize(0, 1) do ε
                -sum(@. log(ε * P_RAND + (1 - ε) * X))
            end
            -opt.minimum
        else
            ε = prod(histogram_size) / (N_SIM_HIST + prod(histogram_size))
            sum(@. log(ε * P_RAND + (1 - ε) * X))
        end
    end
end
