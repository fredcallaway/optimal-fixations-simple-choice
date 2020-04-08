BASE_DIR = "results/test"
SEARCH_STRATEGY = :sobol
GRID_SIZE = 10
FIT_PRIOR = true

SPACE = Box(
    :α => (50, 250),
    :σ_obs => (1, 5),
    :sample_cost => (.001, .01),
    :switch_cost => (.003, .03),
)

# SPACE = Box(
#     :α => (50, 500),
#     :σ_obs => (1, 10),
#     :sample_cost => (.001, .01),
#     :switch_cost => (.003, .03),
# )
# SPACE = Box(
#     :α => (100, 300),
#     :σ_obs => (2, 3.5),
#     :sample_cost => (.001, .006),
#     :switch_cost => (.013, .025),
# )

UCB_PARAMS = (
    N=8,
    n_iter=2,
    n_init=2,
    n_roll=2,
    n_top=2,
)
LIKELIHOOD_PARAMS = (
    fit_ε = true,
    max_ε = 0.5,
    n_sim_hist = 10,
    test_fold = "odd",
    hist_bins = 5
)
