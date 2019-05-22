cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
include("model.jl")
include("job.jl")
include("human.jl")
include("simulations.jl")
include("features.jl")

function load_policy(job)
    m = MetaMDP(job)
    try
        Policy(m, deserialize(job, :optim).θ1)
    catch
        missing
    end
end

using ClusterManagers
using Distributions
elastic_worker("cookie", "10.2.159.72", 58856)