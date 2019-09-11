using Distributed
@everywhere begin
    include("utils.jl")
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("results.jl")
end
using SplitApplyCombine
using StatsBase: countmap
using Serialization

# N = parse(Int, ARGS[1])
m = open(deserialize, "tmp/m")

results = Results("test_ucb_soft")
kws = (N=50^3, α=100., β=3., n_iter=10_000, n_roll=1000, n_init=100, n_top=10)
@show kws

out, t = @timed ucb(m; kws...)
@show t

save(results, :mdp, m)
save(results, :kws, kws)
save(results, :out, out)
save(results, :runtime, t)

# # %% ====================  ====================
# results = map(get_results("test_ucb2")) do res
#     !exists(res, :out) && return missing
#     res
# end |> skipmissing |> collect;

# outs = map(results) do res
#     load(res, :out)
# end

# tops = map(outs) do o
#     argmax(o.μ)
# end

# kws = [load(res, :kws) for res in results]
# β3 = haskey.(kws, :β)

# # %% ====================  ====================


# countmap(tops[β3])
# countmap(tops[.!β3])

# sort(outs[end].μ)

# policies, = ucb(m, N=10000, n_roll=1, n_init=0, n_iter=0)

# policies[9740].θ
# policies[9925].θ
# policies[9572].θ
# # %% ====================  ====================

# runtime = map(results[β3]) do res
#     load(res, :runtime)
# end

# mean(getfield.(outs[β3], :converged))
# mean(getfield.(outs[.!β3], :converged))

# function found_iter(out)
#     top = out.hist.top
#     for i in length(top):-1:1
#         if top[i] != 9740
#             return i
#         end
#     end
# end

# describe(float.(found_iter.(outs[.!β3])))



