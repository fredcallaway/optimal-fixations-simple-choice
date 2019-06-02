cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
include("model.jl")
include("job.jl")
include("human.jl")
include("simulations.jl")
include("features.jl")

using ClusterManagers

function load_policy(job)
    m = MetaMDP(job)
    try
        Policy(m, deserialize(job, :optim).θ1)
    catch
        missing
    end
end

function run_simulations(job::Job)
    pol = load_policy(job)
    ismissing(pol) && return "missing policy"
    for μ in 0:0.1:μ_emp
        if exists(job, :features_*μ)
            continue
        end
        # (prior=(μ, σ_emp), sim=nothing, loss=rand())
        println("I am doing work.")
        sleep(3)
        # sim = simulate_experiment(pol, (µ, σ_emp), 1)
        # serialize(job, :sim_*μ, sim)
        # serialize(job, :prior_*μ, (µ, σ_emp))
        # serialize(job, :features_*μ, compute_features(sim))
    end
    return "success"
end


if ARGS[1] == "worker"
    elastic_worker("cookie", "10.2.159.72", 58856)
else

end

include("master.jl")
include("job.jl")

using Glob
files = glob("runs/rando1000/jobs/*")
jobs = Job.(files)



results = smap(run_simulations, jobs[1:20])
println(results)