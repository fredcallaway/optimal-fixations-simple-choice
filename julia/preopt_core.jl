using Distributed
using Serialization
include("box.jl")
include("meta_mdp.jl")
include("bmps.jl")
include("optimize_bmps.jl")
include("ucb_bmps.jl")
include("human.jl")
include("simulations.jl")

const MAX_STEPS = 200  # 20 seconds
const SAMPLE_TIME = 100

space = Box(
    :α => (100, 300),
    :σ_obs => (1, 5),
    :sample_cost => (.001, .01),
    :switch_cost => (.013, .025),
)

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

x2prm(x) = x |> space |> namedtuple

function get_policies(n_item, prm; n_top=80)
    m = MetaMDP(n_item, prm.σ_obs, prm.sample_cost, prm.switch_cost)
    policies, μ, sem = ucb(m; N=8000, α=prm.α, n_iter=500, n_init=100, n_roll=100, n_top=n_top)
    return policies[partialsortperm(-μ, 1:n_top)]
end

function sim_one(policy, μ, σ, v)
    sim = simulate(policy, (v .- μ) ./ σ; max_steps=MAX_STEPS)
    fixs, fix_times = parse_fixations(sim.samples, SAMPLE_TIME)
    (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times)
end

function simulate_trials(policy::Policy, trials::Table, μ::Float64, σ::Float64)
    map(trials.value) do v
        sim_one(policy, μ, σ, v)
    end |> Table
end


include("pseudo_base.jl")
include("pseudo_likelihood.jl")


function get_loss(policies, ds, β_µ)
    prm = (β_μ=β_μ, β_σ=1., σ_rating=NaN)
    logp, ε, baseline = likelihood(ds, policies, prm; parallel=false);
    logp / baseline
end