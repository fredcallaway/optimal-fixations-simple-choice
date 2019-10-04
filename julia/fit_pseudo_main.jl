include("results.jl")
include("pseudo_likelihood.jl")
include("box.jl")
const results = Results("pseudo_mu_cv")
# const results = Results("pseudo_3_epsilon")


μ_space = Dict(
    "fit" => (0, μ_emp),
    "emp" => μ_emp,
)[ARGS[1]]
index = Dict(
    "odd" => 1:2:length(trials),
    "even" => 2:2:length(trials),
)[ARGS[2]]

space = Box(
    :α => (50, 1000, :log),
    :σ_obs => (1, 10),
    :sample_cost => (1e-3, 5e-2, :log),
    :switch_cost => (1e-3, 1e-1, :log),
    # :µ => μ_emp,
    :σ => σ_emp,
    :sample_time => 100,
    :µ => μ_space,
    # :σ => (σ_emp / 4, 4 * σ_emp),
)

opt_kws = (
    iterations=100,
    init_iters=100,
    acquisition="ei",
    optimize_every=5,
    acquisition_restarts=200,
    noisebounds=[-4, 1],
)
like_kws = (
    index = 1:2:length(trials),
    fit_ε = true,
)

save(results, :opt_kws, opt_kws)
save(results, :like_kws, like_kws)
save(results, :metrics, the_metrics)
save(results, :space, space)

loss_iter = 1
function loss(prm::Params; kws...)
    m = MetaMDP(prm)
    policy = optimize_bmps(m; α=prm.α, kws...)
    likelihood = total_likelihood(policy, prm; like_kws...)
    save(results, Symbol(string("loss_", lpad(loss_iter, 3, "0"))),
         (prm=prm, policy=policy, likelihood=likelihood);
         verbose=false)
    global loss_iter += 1
    baseline = length(like_kws.index) * log(P_RAND)
    isfinite(likelihood) ? min(likelihood / baseline, 10) : 10
end

function loss(x::Vector{Float64}; kws...)
    # println(round.(x; digits=4))
    loss(Params(space(x)); kws...)
end

function fit(opt)
    for i in 1:8
        boptimize!(opt)
        find_model_max!(opt)
        prm = opt.model_optimizer |> space |> Params
        save(results, Symbol(string("mle_", loss_iter)), prm)
        @info "Iteration $loss_iter" prm opt.model_optimum
    end
    opt.model_optimizer |> space |> Params
end

function reoptimize(prm::Params; N=16)
    policies = asyncmap(1:N) do i
        m = MetaMDP(prm)
        optimize_bmps(m; α=prm.α)
    end
    save(results, :reopt, policies)
end

opt = gp_minimize(loss, n_free(space); run=false, opt_kws...)
mle = fit(opt)
reoptimize(mle)
save(results, :opt, opt)
