BASE_DIR = "results/rando"
SEARCH_STRATEGY = :sobol
GRID_SIZE = -1
FIT_PRIOR = false

SPACE = Box(
    :σ_obs => (1, 5),
    :p_switch => (0, 0.5),
    :p_stop => (0, 0.2),
    :sample_cost => NaN,
    :switch_cost => NaN,

)

LIKELIHOOD_PARAMS = (
    fit_ε = true,
    max_ε = 0.5,
    n_sim_hist = 10_000,
    test_fold = "odd",
    hist_bins = 5
)

struct Rando <: Policy
    m::MetaMDP
    p_switch::Float64
    p_stop::Float64
end

function (pol::Rando)(b::Belief)
    rand() < pol.p_stop && return ⊥
    b.focused == 0 && return rand(1:pol.m.n_arm)
    rand() > pol.p_switch && return b.focused
    return rand(setdiff(1:pol.m.n_arm, b.focused))
end
