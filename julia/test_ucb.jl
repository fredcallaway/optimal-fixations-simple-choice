using Distributed
@everywhere begin
    include("utils.jl")
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("ucb_bmps.jl")
    include("results.jl")
    using Serialization
    m = open(deserialize, "tmp/m")
end

@everywhere function run_one(;kws...)
    results = Results("test_ucb_sep17")
    out, t = @timed optimize_bmps_ucb(m; kws...)
    save(results, :mdp, m; verbose=false)
    save(results, :kws, values(kws); verbose=false)
    save(results, :runtime, t; verbose=false)
    save(results, :out, out)

    # rank = sortperm(μ, rev=true)
    # top = rank[1:10]
    # policies = all_policies[top]
    # all_policies, μ = out
    # policies = all_policies[partialsortperm(μ, 1:10; rev=true)]
    # save(results, :likelihood, total_likelihood(policies))
end

include("box.jl")

args = [(N=N, α=α, β=3., n_iter=n_iter, n_roll=1000, n_init=100, n_top=n_top)
         for α in (200,)
         for N in (20^3, 40^3)
         for n_top in (1,)
         for n_iter in (1000, 5000)]


map(repeat(args, 30)) do arg
    run_one(;arg...)
end

if false # analyzing results

# %% ====================  ====================
using TypedTables
using SplitApplyCombine
results = filter(get_results("test_ucb_sep17")) do res
    exists(res, :out)
end

# %% ====================  ====================
pol = load(results[1], :out)[1]
bmps_policies = asyncmap(1:8) do i
    optimize_bmps(pol.m; α=pol.α)
end
bmps_rs = map(bmps_policies) do pol
    mean_reward(pol, 100_000, true)
end

# %% ====================  ====================
T = map(results) do res
    pol, r = load(res, :out)
    # r1 = mean_reward(pol, 100_000, true)
    r1 = 0
    (pol.θ..., r=r, r1=r1, time=load(res, :runtime), load(res, :kws)...)
    # (kws..., pol.θ..., r=r)
end |> FlexTable

# %% ====================  ====================

group(t->(t.n_iter, t.N), t->t.r1, T) |> valmap(juxt(mean, std)) |> sort




# %% ====================  ====================

end

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



