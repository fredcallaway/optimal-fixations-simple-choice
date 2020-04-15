BASE_DIR = "results/grid13"
SEARCH_STRATEGY = :grid
GRID_SIZE = 7
FIT_PRIOR = false

# SPACE = Box(
#     :α => (100, 350),
#     :σ_obs => (1, 5),
#     :sample_cost => (.001, .01),
#     :switch_cost => (.003, .03),
# )

SPACE = Box(
    :α => (100, 500),
    :σ_obs => (1, 5),
    :sample_cost => (.0001, .01),
    :switch_cost => (.003, .05),
)

UCB_PARAMS = (
    N = 20^3,
    n_iter=5000,
    n_init=100,
    n_roll=10,
    n_top=80
)
LIKELIHOOD_PARAMS = (
    fit_ε = true,
    max_ε = 0.5,
    n_sim_hist = 50_000,
    test_fold = "odd",
    hist_bins = 5
)