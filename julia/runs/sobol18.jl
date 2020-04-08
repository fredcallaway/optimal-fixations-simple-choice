BASE_DIR = "results/sobol18"
SEARCH_STRATEGY = :sobol
GRID_SIZE = 7
FIT_PRIOR = true

SPACE = Box(
    :α => (100, 350),
    :σ_obs => (1, 5),
    :sample_cost => (.001, .01),
    :switch_cost => (.003, .03),
)

UCB_PARAMS = (
    n_iter=500,
    n_init=100,
    n_roll=100,
    n_top=80
)
LIKELIHOOD_PARAMS = (
    fit_ε = true,
    max_ε = 0.5,
    n_sim_hist = 10_000,
    test_fold = "odd",
    hist_bins = 5
)
