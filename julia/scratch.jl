include("model.jl")

include("optimize.jl")

prm = Params("runs/dingo/jobs/1.json")
pol = bmps_policy(Array{Float64}(load(prm, :optim)["theta"]))

m = Pomdp(prm)
b = Belief(m)
