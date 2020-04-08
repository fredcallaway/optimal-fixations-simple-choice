BASE_DIR = "results/attention_noise"
SEARCH_STRATEGY = :sobol
GRID_SIZE = 6
FIT_PRIOR = true
(α = 242.9, σ_obs = 3.357, sample_cost = 0.005223, switch_cost = 0.009974, attention_noise = 0.06665, β_μ = 0.4)
 (α = 242.9, σ_obs = 3.357, sample_cost = 0.005223, switch_cost = 0.009974, attention_noise = 0.06665, β_μ = 0.6)
 (α = 243.2, σ_obs = 2.152, sample_cost = 0.003101, switch_cost = 0.01146, attention_noise = 0.1343, β_μ = 0.4)
 (α = 243.2, σ_obs = 2.152, sample_cost = 0.003101, switch_cost = 0.01146, attention_noise = 0.1343, β_μ = 0.8)
 (α = 243.2, σ_obs = 2.152, sample_cost = 0.003101, switch_cost = 0.01146, attention_noise = 0.1343, β_μ = 0.6)
 (α = 242.9, σ_obs = 3.357, sample_cost = 0.005223, switch_cost = 0.009974, attention_noise = 0.06665, β_μ = 0.8)
 (α = 248.5, σ_obs = 2.479, sample_cost = 0.004529, switch_cost = 0.01058, attention_noise = 0.3425, β_μ = 0.8)
 (α = 169.9, σ_obs = 3.117, sample_cost = 0.004006, switch_cost = 0.01349, attention_noise = 0.03027, β_μ = 0.6)

SPACE = Box(
    :α => (150, 300),
    :σ_obs => (2, 4),
    :sample_cost => (.003, .006),
    :switch_cost => (.01, .02),
    :attention_noise => (0.05, 0.25)
)

UCB_PARAMS = (
    n_iter=500,
    n_init=100,
    n_roll=50,
    n_top=80
)

LIKELIHOOD_PARAMS = (
    fit_ε = true,
    max_ε = 0.5,
    n_sim_hist = 10_000,
    test_fold = "odd",
    hist_bins = 5
)
