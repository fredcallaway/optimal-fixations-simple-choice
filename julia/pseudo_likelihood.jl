using Distributed

@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("human.jl")
    include("binning.jl")
    include("simulations.jl")
    include("features.jl")
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

    struct Metric{F}
        f::F
        bins::Binning
    end

    Metric(f::Function, n::Int) = Metric(f, Binning(f.(trials), n))
    (m::Metric)(t) = t |> m.f |> m.bins

    m = Metric(total_fix_time, 10)
    counts(m.(trials))  # something's wrong?

    const the_metrics = [
        Metric(total_fix_time, 10),
        Metric(n_fix, Binning([1:7; Inf])),
        Metric(rank_chosen, Binning(1:4)),
        Metric(chosen_fix_proportion, 2)
    ]

    const apply_metrics = juxt(the_metrics...)
    const max_steps = Int(fld(the_metrics[1].bins.limits[end], 100))
    const histogram_size = Tuple(length(m.bins) for m in the_metrics)

    function sim_one(policy, v)
        μ, σ = μ_emp, σ_emp
        sim = simulate(policy, (v .- μ) ./ σ; max_steps=max_steps)
        fixs, fix_times = parse_fixations(sim.samples, 100)
        (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times)
    end

    function get_metrics(policy, v, N)
        ms = @distributed vcat for i in 1:N
            sim = sim_one(policy, v)
            n_fix(sim) == 0 ? missing : apply_metrics(sim)
        end
        skipmissing(ms)
    end
    function likelihood_matrix(policy, v::Vector{Float64}; N=N_SIM_HIST)
        L = zeros(histogram_size...)
        for m in get_metrics(policy, v, N)
            L[m...] += 1
        end
        L ./ sum(L)
    end
end

# %% ====================  ====================
const P_RAND = 1 / prod(histogram_size)
const BASELINE = log(P_RAND) * length(trials)

function total_likelihood(policies)
    vs = unique(sort(t.value) for t in trials);
    sort!(vs, by=std)  # fastest trials last for parallel efficiency
    args = collect(Iterators.product(policies, vs));
    out = asyncmap(args) do (policy, v)
        likelihood_matrix(policy, v)
    end

    likelihoods = Dict(zip(args, out))
    function likelihood(policy, t::Trial)
        L = likelihoods[policy, sort(t.value)]
        L[apply_metrics(t)...]
    end

    ε = prod(histogram_size) / (N_SIM_HIST + prod(histogram_size))
    X = likelihood.(reshape(policies, (1, :)), trials);
    sum(@. log(ε * P_RAND + (1 - ε) * X); dims=1)[:]
end
