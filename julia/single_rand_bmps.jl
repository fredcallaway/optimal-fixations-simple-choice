include("results.jl")
results = Results("aug19")
include("bmps_moments_fitting.jl")

# using Logging; global_logger(SimpleLogger(stdout, Logging.Debug))

x = rand(3)
prm = Params(space(x))
save(results, :prm, prm)
m = MetaMDP(prm)
@time policies, fitness, n = halving(m)
rnk = sortperm(-fitness)
@time top = map(1:30) do j
    i = rnk[j]
    pol = policies[i]
    sim = simulate_experiment(pol, 100)
    (fit=fitness[i], pol=pol, losses=multi_loss(sim))
end
save(results, :top, top)




