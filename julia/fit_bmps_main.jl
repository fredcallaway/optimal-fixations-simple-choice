using Distributed
addprocs()
include("bmps_moments_fitting.jl")

# %% ====================  ====================
using Glob

X, y = let
    xs, y = asyncmap(glob("results/halving/bmps/rand/*/record")) do f
        record = open(deserialize, f)
        invert((record.x, record.y))
    end |> flatten |> invert
    X = combinedims(xs)
    (X, y)
end

# %% ====================  ====================
policies, y = let
    asyncmap(glob("results/halving/bmps/rand/*/record")) do f
        record = open(deserialize, f)
        invert((record.policies, record.y))
    end |> flatten |> invert
end

policies[argmin(y)]


# %% ==================== Check close-to-optimal policies ====================
min_y, i = findmin(y)
x = X[:, i]
prm = Params(space(x))
m = MetaMDP(prm)
@time policies, fitness, n = halving(m)
rnk = sortperm(-fitness)

using Printf

for i in rnk[1:10]
    pol = policies[i]
    losses = map(1:30) do i
        sim = simulate_experiment(pol, 10)
        sim_loss(sim)
    end
    @printf "%.5f %.5f\n" mean(losses) std(losses)
end


# %% ====================  ====================
opt = gp_minimize(loss, n_free(space),
    acquisition_restarts=200,
    noisebounds=[-4, 1],
    iterations=200,
    optimize_every=5,
    run=false,
    acquisition="ei",
    init_Xy=(X, y)
)
optimize!(opt.model)
meanvar(x) = BayesianOptimization.mean_var(opt.model, x)
find_model_max!(opt)

# fx = loss(opt.model_optimizer)

@async boptimize!(opt)

# %% ====================  ====================
open("tmp/best_bmps_sim", "w+") do f
    i = argmin(RECORD.y)
    policy = RECORD.policies[i]
    serialize(f, simulate_experiment(policy, 100))
end

# %% ====================  ====================
X, y = init_Xy
x1 = X[:, argmin(y)]

@async begin
    policy = bmps_policy(x1)
    open("tmp/best_bmps_sim", "w+") do f
        serialize(f, simulate_experiment(policy, 100))
    end
end


# %% ====================  ====================
x = [0.057, 0.74, 0.075]
m = MetaMDP(Params(space(x)))
@time results = asyncmap(1:10) do i
    optimize_bmps(m, n_roll=5000, n_iter=200)
end

# %% ====================  ====================
pols, opts = invert(results);

rewards = asyncmap(pols) do p
    mean_reward(p, 10000, true)
end

for p in pols
    println(round.(collect(p.θ), digits=3))
    mn, vr = mean_var(opt.model, opt.observed_optimizer)
    @printf "  %.3f  %.0e  " mn √vr
end

# %% ====================  ====================



# bmps_policy(x::Vector{Float64}) = bmps_policy(MetaMDP(Params(space(x))))
# prior(prm::Params) = (prm.μ, prm.σ)

# prepare_result(prm::Params) = (
#     policy = bmps_policy(MetaMDP(prm)),
#     prior = prior(prm),
#     sample_time = prm.sample_time
# )



function save_results()
    println("observed: ", round.(opt.observed_optimizer; digits=3),
            " => ", round(opt.observed_optimum; digits=5))
    println("model:    ", round.(opt.model_optimizer; digits=3),
            " => ", round(opt.model_optimum; digits=5))
    f_mod = @show loss(opt.model_optimizer)
    f_obs = @show loss(opt.observed_optimizer)
    best_x = f_obs < f_mod ? opt.observed_optimizer : opt.model_optimizer
    prm = Params(space(best_x))

    println("Best fitting optimal policy:")
    println(bmps_policy(MetaMDP(prm)))

    save(results, :opt, opt)
    save(results, :model, opt.model)
    save(results, :xy, (x=opt.model.x, y=opt.model.y))
    save(results, :best, prepare_result(prm))

    policy, opt = optimize_bmps(MetaMDP(prm))
    println("Reoptimized policy")
    println(policy.θ)
    save(results, :opt_again, (policy, opt.model))
end

save_results()

for i in 1:10
    boptimize!(opt)
    save_results()
end