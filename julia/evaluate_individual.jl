include("fit_base.jl")
include("compute_policies.jl")
include("human.jl")
include("simulations.jl")
include("plots_preprocessing.jl")

path = "$BASE_DIR/individual"
all_top = deserialize("$path/all_top")
K = deserialize("$path/K")

function recompute_policies(job::Int)
    n_item = K[job][1]
    compute_policies(n_item, all_top[job]; UCB_PARAMS...)
end

function compute_simulations(job::Int)
    policies = deserialize("$path/test_policies/$job")
    prm = all_top[job]
    n_item, subject = K[job]
    prior = make_prior(load_dataset(n_item), prm.β_μ)
    trials = filter(load_dataset(n_item, :test)) do t
        t.subject == subjectt
    end
    map(policies) do pol
        simulate_trials(pol, prior, trials)
    end
end

function compute_plot_features(job::Int)
    trials = reduce(vcat, deserialize("$path/simulations/$job"))
    compute_plot_features(trials)
end

function do_all(job::Int)
    # do_job(recompute_policies, "individual/test_policies", job)
    # do_job(compute_simulations, "individual/simulations", job)
    do_job(compute_plot_features, "individual/plot_features", job)
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    job = parse(Int, ARGS[1])
    do_all(job)
end
