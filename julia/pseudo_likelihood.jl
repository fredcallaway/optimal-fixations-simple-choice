using Distributed

@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("human.jl")
    include("binning.jl")
    include("simulations.jl")
end

total_fix_time(t::Trial) = sum(t.fix_times)
tft_bins = Binning(total_fix_time.(trials), 10)
n_fix_bins = Binning([1:7; Inf])

@everywhere begin
    μ_emp = $μ_emp
    σ_emp = $σ_emp
    tft_bins = $tft_bins
    n_fix_bins = $n_fix_bins

    function sim_one(policy, v)
        sim = simulate(policy, (v .- μ_emp) ./ σ_emp)
        fixs, fix_times = parse_fixations(sim.samples, 100)
        (total_fix_time=sum(fix_times),
         rank_chosen=sortperm(v; rev=true)[sim.choice],
         n_fix=length(fixs))
    end

    function sim_many(policy, v; N=10_000)
        map(1:N) do i
            sim_one(policy, v)
        end
    end

    function metrics(t::Trial)
        (total_fix_time=sum(t.fix_times),
         rank_chosen=sortperm(t.value; rev=true)[t.choice],
         n_fix=length(t.fixations))
    end

    function index(s)
        tft_bins(s.total_fix_time), s.rank_chosen, n_fix_bins(s.n_fix)
    end

    function likelihood_matrix(policy, v::Vector{Float64})
        L = zeros(10, 3, 7)
        for s in sim_many(policy, v)
            s.n_fix == 0 && continue
            L[index(s)...] += 1
        end
        L ./ sum(L)
    end
end

const P_RAND = 1 / (10 * 3 * 7)
const BASELINE = log(P_RAND) * length(trials)

function total_likelihood(policies)
    vs = unique(sort(t.value) for t in trials);
    sort!(vs, by=std)  # fastest trials last for parallel efficiency
    args = collect(Iterators.product(policies, vs));
    out = pmap(args) do (policy, v)
        likelihood_matrix(policy, v)
    end

    likelihoods = Dict(zip(args, out))
    function likelihood(policy, t::Trial)
        L = likelihoods[policy, sort(t.value)]
        L[index(metrics(t))...]
    end

    ε = 10 * 3 * 7 / (10000 + 10 * 3 * 7)
    X = likelihood.(reshape(policies, (1, :)), trials);
    sum(@. log(ε * P_RAND + (1 - ε) * X); dims=1)[:]
end



