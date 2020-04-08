using Pkg
Pkg.add(split("Glob Distributions Memoize Parameters SplitApplyCombine StaticArrays TypedTables StatsBase Sobol DataStructures Optim QuadGK ArgParse StatsFuns OnlineStats"))

# Precompile
include("fit_base.jl")
include("compute_policies.jl")
include("compute_likelihood.jl")
