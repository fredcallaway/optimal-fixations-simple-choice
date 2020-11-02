include("fit_base.jl")
job = parse(Int, ARGS[1])
out = "$BASE_DIR/recovery/sims/$job"
if isfile(out)
    println("Job already completed!")
    exit(0)
end

include("box.jl")
include("meta_mdp.jl")
include("bmps.jl")
include("bmps_ucb.jl")
include("human.jl")
include("simulations.jl")

using Sobol
mkpath("$BASE_DIR/recovery/sims")
space = Box(
    :α => (100, 500),
    :σ_obs => (1, 5),
    :sample_cost => (.001, .01),
    :switch_cost => (.003, .03),
    :β_μ => (0, 1)
)

seq = SobolSeq(n_free(space))
skip(seq, job-1; exact=true)
prm = next!(seq) |> space |> namedtuple

if job == 1
    seq = SobolSeq(n_free(space))
    true_prms = [namedtuple(space(next!(seq))) for i in 1:1024]
    serialize("$BASE_DIR/recovery/true_prms", true_prms)
end

sims = map([2, 3]) do n_item
    m = MetaMDP(n_item, prm)
    all_policies, μ, sem = ucb_policies(m; α=prm.α, UCB_PARAMS...)
    policies = all_policies[partialsortperm(-μ, 1:UCB_PARAMS.n_top)]

    all_trials = load_dataset(n_item);
    prior = make_prior(all_trials, prm.β_μ)
    trials = filter(all_trials) do t
        t.trial % 2 == 0
    end

    map(trials.value, Iterators.cycle(policies)) do v, policy
        simulate(policy, prior, v)
    end |> Table
end

serialize(out, (prm=prm, sims=sims))

