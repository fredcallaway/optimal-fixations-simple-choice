using Serialization
include("model_base.jl")
include("box.jl")
include("optimize_bmps.jl")

const space = Box(
    :obs_sigma => (1, 10),
    # :sample_cost => (1e-4, 1e-2, :log),
    :sample_cost => (1e-3, 1e-2, :log),
    :switch_cost => (1, 60),
)

const NAME = "foobar"
const N_RAND = 1000
const N_ITER = 800
const N_ROLL = 1000

# using Sobol
# sobol_points = let
#     seq = SobolSeq(3)
#     skip(seq, N_RAND)
#     collect(Iterators.take(seq, N_RAND))
# end
# open("sobol1000", "w+") do f
#     serialize(f, sobol_points)
# end

# x = length(ARGS) > 0 ? open(deserialize, "sobol1000")[parse(Int, ARGS[1])] : rand(3)

x = rand(3)
results = Results(NAME)
mdp = MetaMDP(;space(x)...)
println(mdp)

save(results, :mdp, mdp; verbose=false)
@time policy, opt = optimize(mdp; n_iter=N_ITER, n_roll=N_ROLL, parallel=false, verbose=true)
save(results, :policy, policy; verbose=false)
save(results, :opt, opt; verbose=false)

observed = round.(opt.observed_optimizer; digits=3) => round(opt.observed_optimum; digits=5)
model = round.(opt.model_optimizer; digits=3) => round(opt.model_optimum; digits=5)
println(string(
    "────────────────────────────────────",
    "\nobserved: ", observed,
    "\nmodel:    ", model
))

