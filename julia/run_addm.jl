include("plots_preprocessing.jl")
include("addm.jl")
mkpath("results/addm")
N_REPEAT = 500
USE_SEM = true

@time sims = map([2, 3]) do n_item
    trials = repeat(load_dataset(n_item, :test), N_REPEAT)
    simulate_trials(ADDM(), trials)
end
serialize("results/addm/sims", sims)

@time pf = compute_plot_features(sims)
serialize("results/addm/plot_features", pf)
