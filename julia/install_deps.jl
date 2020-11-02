using Pkg
println("Installing packages...")
Pkg.add(split("Glob Distributions Memoize Parameters SplitApplyCombine StaticArrays TypedTables StatsBase Sobol DataStructures Optim QuadGK ArgParse StatsFuns OnlineStats Bootstrap"))
Pkg.update("OnlineStats")
println("Precompiling...")
include("fit_base.jl")
include("compute_policies.jl")
include("compute_likelihood.jl")
include("evaluation.jl")
