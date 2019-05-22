cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
include("model.jl")
include("job.jl")
include("human.jl")
include("simulations.jl")
include("features.jl")
include("elastic.jl")

function load_policy(job)
    m = MetaMDP(job)
    try
        Policy(m, deserialize(job, :optim).θ1)
    catch
        missing
    end
end

using Glob
files = glob("runs/rando1000/jobs/*")
jobs = Job.(files)

function run_simulations(i)
    job = jobs[i]
    pol = load_policy(job)
    if ismissing(pol)
        println("job $i: missing policy")
        return
    end
    if exists(job, :features)
        println("job $i: already completed")
        return
    end
    features = map(0:0.1:μ_emp) do μ_emp
        if exists(job, :sim_*μ)
            sim = deserialize(job, :sim_*μ)
        else
            sim = simulate_experiment(pol, (µ, σ_emp), 1)
            serialize(job, :sim_*μ, sim, verbose=false)
        end
        compute_features(sim)
    end
    serialize(job, :features, features, verbose=false)
    println("job $i: success")
end

try
    start_worker()
catch
    start_master()
    results = smap(run_simulations, 1:length(jobs))
    println(results)
end
