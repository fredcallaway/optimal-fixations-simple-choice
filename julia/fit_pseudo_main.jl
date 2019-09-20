include("pseudo_likelihood.jl")
include("box.jl")
include("results.jl")
include("params.jl")
const RESULT_NAME = "fit_pseudo_3"

space = Box(
    :α => (50, 1000, :log),
    :σ_obs => (1, 10),
    :sample_cost => (1e-3, 5e-2, :log),
    :switch_cost => (1e-3, 1e-1, :log),
    :µ => μ_emp,
    :σ => σ_emp,
    :sample_time => 100
    # (0, 2 * μ_emp),
    # :σ => (σ_emp / 4, 4 * σ_emp),
)

function loss(prm::Params; kws...)
    m = MetaMDP(prm)
    policy = optimize_bmps(m; α=prm.α, kws...)[1]
    likelihood = total_likelihood([policy])[1]
    # println("  => ", round(likelihood))

    res = Results(RESULT_NAME)
    save(res, :out, (prm=prm, policy=policy, likelihood=likelihood); verbose=false)
    min(likelihood / BASELINE, 10)
end

function loss(x::Vector{Float64}; kws...)
    # println(round.(x; digits=4))
    loss(Params(space(x)); kws...)
end

opt = gp_minimize(loss, n_free(space),
    acquisition_restarts=200,
    noisebounds=[-4, 1],
    iterations=100,
    optimize_every=5,
    run=true,
    acquisition="ei",
    # init_Xy=(X, y)
)

for i in 2:4
    boptimize!(opt)
    open("tmp/$(RESULT_NAME)_model_mle", "w") do f
        prm = opt.model_optimizer |> space |> Params
        serialize(f, prm)
    end

    model_loss = loss(opt.model_optimizer)
    model_pol = load(get_results(RESULT_NAME)[end], :out)
    open("tmp/$(RESULT_NAME)_policy_$(i)00", "w") do f
        serialize(f, model_pol)
    end
end


open("tmp/$(RESULT_NAME)_model_mle", "w") do f
    prm = opt.model_optimizer |> space |> Params
    serialize(f, prm)
end


model_loss = loss(opt.model_optimizer)
model_mle = load(get_results(RESULT_NAME)[end], :out)
open("tmp/$(RESULT_NAME)_model_mle", "w") do f
    serialize(f, model_mle)
end

# meanvar(x) = BayesianOptimization.mean_var(opt.model, x)


# for i in 1:100
#     println(loss([0.5, 0.5, 0.5]; n_roll=10_000))
# end

# # using Glob
# # outs = map(glob("results/bmps_moments/*/out")) do f
# #     open(deserialize, f)
# # end

# # o = outs[1]


# likes = map(outs) do o
#     like = total_likelihood([o.policy])[1]
#     println(round.(collect(o.policy.θ); digits=3), "   ",
#         round(o.likelihood[1]), "  ",
#         round(like))
#     o.likelihood[1], like
# end
