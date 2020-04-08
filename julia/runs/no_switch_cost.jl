BASE_DIR = "results/no_switch_cost"
SEARCH_STRATEGY = :sobol
GRID_SIZE = 10
FIT_PRIOR = false

SPACE = Box(
    :α => (1, 250),
    :σ_obs => (1, 5),
    :sample_cost => (.001, .02),
    :switch_cost => 0.,
)

UCB_PARAMS = (
    n_iter=500,
    n_init=30,
    n_roll=10,
    n_top=80
)
LIKELIHOOD_PARAMS = (
    fit_ε = true,
    max_ε = 0.5,
    n_sim_hist = 10_000,
    test_fold = "odd",
    hist_bins = 5
)

LESION_ATTENTION = true