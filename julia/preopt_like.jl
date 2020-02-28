ucb_path = "results/grid/ucb"
out = "results/grid/like"
mkpath(out)


include("pseudo_likelihood.jl")
include("pseudo_base.jl")
args = Dict(
    "hist_bins" => 5,
    "propfix" => true,
    "fold" => "odd",
)

like_kws = (
    fit_ε = true,
    max_ε = 0.5,
    n_sim_hist = 10_000
)
n_top = 80



function compute_likelihood(job)
    dest = "$out/$job"
    if isfile(dest)
        println("Job $job has been completed")
        return
    elseif isfile(dest * "x")
        println("Job $job is in progress")
        return
    end
    touch(dest*"x")
    println("Computing likelihood for job ", job)

    datasets = [build_dataset("two", -1), build_dataset("three", -1)]
    ucb = deserialize("$ucb_path/$job")
    results = map(0:0.1:1) do β_μ
        losses = map(1:2) do item_idx
            policies, μ, sem = ucb[item_idx]
            ds = datasets[item_idx]
            top = policies[partialsortperm(-μ, 1:n_top)]
            prm = (β_μ=β_μ, β_σ=1., σ_rating=NaN)
            logp, ε, baseline = likelihood(ds, top, prm; parallel=false);
            logp / baseline
        end
        pol = ucb[1][1][1];
        prm = (namedtuple(type2dict(pol.m))..., α=pol.α, β_μ=β_μ)
        prm, losses
    end

    serialize(dest, results)
    isfile(test*"x") && rm(dest*"x")
    println("Wrote $dest")
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    job = parse(Int, ARGS[1])
    compute_likelihood(job)
end