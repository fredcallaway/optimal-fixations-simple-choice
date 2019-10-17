include("results.jl")
include("pseudo_likelihood.jl")
include("box.jl")
const results = Results("new_pseudo")
# const results = Results("pseudo_3_epsilon")


μ_space = Dict(
    "fit" => (0, μ_emp),
    "emp" => μ_emp,
)[ARGS[1]]

fit_ε = Dict(
    "fit" => true,
    "one" => false,
)[ARGS[2]]

index = Dict(
    "odd" => 1:2:length(trials),
    "even" => 2:2:length(trials),
    "all" => 1:length(trials),
)[ARGS[3]]

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
    index = index,
    fit_ε = fit_ε,
    max_ε = 0.5
)

save(results, :opt_kws, opt_kws)
save(results, :like_kws, like_kws)
save(results, :metrics, the_metrics)
save(results, :space, space)

@info "Begin fitting" opt_kws like_kws

loss_iter = 1
function loss(prm::Params; kws...)
    m = MetaMDP(prm)
    policy = optimize_bmps(m; α=prm.α, kws...)
    likelihood, ε, baseline = total_likelihood(policy, prm; like_kws...)
    save(results, Symbol(string("loss_", lpad(loss_iter, 3, "0"))),
         (prm=prm, policy=policy, ε=ε, likelihood=likelihood);
         verbose=false)
    global loss_iter += 1

    max_loss = 2
    ll = isfinite(likelihood) ? min(likelihood / baseline, max_loss) : max_loss
    @printf "%.2f   %.4f\n" ε ll
    ll
end

function loss(x::Vector{Float64}; kws...)
    print("($loss_iter)  ", round.(x; digits=3), "  =>  ")
    loss(Params(space(x)); kws...)
end

function fit(opt)
    for i in 1:4
        boptimize!(opt)
        find_model_max!(opt)
        prm = opt.model_optimizer |> space |> Params
        save(results, Symbol(string("mle_", loss_iter)), prm)
        @info "Iteration $loss_iter" prm opt
    end
end

function reoptimize(prm::Params; N=16)
    policies = asyncmap(1:N) do i
        m = MetaMDP(prm)
        optimize_bmps(m; α=prm.α)
    end
    save(results, :reopt, policies)

    reopt_like = asyncmap(policies) do policy
        total_likelihood(policy, prm; like_kws...)
    end
    save(results, :reopt_like, reopt_like)

    test_index = setdiff(eachindex(trials), like_kws.index)
    test_kws = (like_kws..., index=test_index)
    test_like = asyncmap(policies) do policy
        total_likelihood(policy, prm; test_kws...)
    end
    save(results, :test_like, test_like)
end

function test_like(res)
    test_index = setdiff(eachindex(trials), like_kws.index)
    like_kws = (like_kws..., index=test_index)
    map(policies) do pol
        total_likelihood(pol, mle; like_kws...)[1:3]
    end

opt = gp_minimize(loss, n_free(space); run=false, verbose=false, opt_kws...)

fit(opt)
find_model_max!(opt)
mle = opt.model_optimizer |> space |> Params
save(results, :mle, mle)
save(results, :opt, opt)
reoptimize(mle)
