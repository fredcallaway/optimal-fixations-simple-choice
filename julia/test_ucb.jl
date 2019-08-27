include("utils.jl")
include("meta_mdp.jl")
include("bmps.jl")
include("optimize_bmps.jl")
include("results.jl")
using SplitApplyCombine
using StatsBase: countmap
using Serialization

# N = parse(Int, ARGS[1])
N = 10000
@show N
m = open(deserialize, "tmp/m")

results = Results("test_ucb2")
kws = (N=N, n_iter=2^20, n_roll=64, n_init=4, β=3.)
@show kws

out, t = @timed ucb(m; kws...)
@show t

save(results, :mdp, m)
save(results, :kws, kws)
save(results, :out, out)
save(results, :runtime, t)