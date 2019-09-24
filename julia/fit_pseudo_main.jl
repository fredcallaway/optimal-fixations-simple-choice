include("results.jl")
include("pseudo_likelihood.jl")
include("box.jl")
include("params.jl")
const results = Results("pseudo_3")

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

save(results, :metrics, the_metrics)
save(results, :space, space)

loss_iter = 1

function loss(prm::Params; kws...)
    m = MetaMDP(prm)
    policy = optimize_bmps(m; α=prm.α, kws...)
    likelihood = total_likelihood([policy])[1]
    save(results, Symbol(string("loss_", lpad(loss_iter, 3, "0"))),
         (prm=prm, policy=policy, likelihood=likelihood);
         verbose=false)
    global loss_iter += 1
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
    init_iters=100,
    optimize_every=5,
    run=false,
    acquisition="ei",
    # init_Xy=(X, y)
)

function fit()
    for i in 1:4
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

mle = fit()
reoptimize(mle)
