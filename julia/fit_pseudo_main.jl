include("pseudo_likelihood.jl")
include("box.jl")
include("results.jl")

const RESULT_NAME = "fit_pseudo_gp"

@with_kw mutable struct Params
    α::Float64
    σ_obs::Float64
    sample_cost::Float64
    switch_cost::Float64
    µ::Float64
    σ::Float64
    sample_time::Float64
end
Params(d::AbstractDict) = Params(;d...)

MetaMDP(prm::Params) = MetaMDP(
    3,
    prm.σ_obs,
    prm.sample_cost,
    prm.switch_cost,
)

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
    @time policy = optimize_bmps(m; α=prm.α, kws...)[1]
    @time likelihood = total_likelihood([policy])[1]
    println("  => ", round(likelihood))

    res = Results(RESULT_NAME)
    save(res, :out, (prm=prm, policy=policy, likelihood=likelihood); verbose=false)
    min(likelihood / BASELINE, 10)
end

function loss(x::Vector{Float64}; kws...)
    println(round.(x; digits=4))
    loss(Params(space(x)); kws...)
end


function prm2x(prm::Params)
    prm_keys = [:α, :σ_obs, :sample_cost, :switch_cost, :µ, :σ, :sample_time]
    d = Dict(k => getfield(prm, k) for k in prm_keys)
    for k in prm_keys
        dim = space.dims[k]
        if length(dim) == 1
            @assert d[k] ≈ dim
        end
        # else
        #     if !(dim[1] <= d[k] <= dim[2])
        #         println(dim[1], " <= ", d[k] ," <= ", dim[2])
        #     end
        #     @assert dim[1] <= d[k] <= dim[2]
        # end
    end
    space(d)
end

xs, y = map(results) do res
    out = load(res, :out)
    x = prm2x(out.prm)
    y = min(out.likelihood / BASELINE, 10)
    x, y
end |> invert
X = combinedims(xs)


opt = gp_minimize(loss, n_free(space),
    acquisition_restarts=200,
    noisebounds=[-4, 1],
    iterations=400,
    optimize_every=5,
    run=false,
    acquisition="ei",
    init_Xy=(X, y)
)
boptimize!(opt)


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
