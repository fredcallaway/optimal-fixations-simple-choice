using Distributed
using SplitApplyCombine
# cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
@everywhere include("simulations.jl")
@everywhere include("loss.jl")

@everywhere function optimize_prior(job)
    pol = optimized_policy(job)
    if ismissing(pol)
        return
    end
    candidates = collect(0:0.2:μ_emp)
    push!(candidates, μ_emp)
    sims, losses = map(candidates) do μ
        sim = simulate_experiment(pol, (μ, σ_emp), 1)
        l = loss(sim)
        (sim, l)
    end |> invert
    serialize(job, :sims_and_loss, (sims, losses))
    best = argmin(losses)
    μ_opt = candidates[best]
    opt = (μ_opt, σ_emp), losses[best]
    serialize(job, :optimized_prior, opt)
end

# const pol = optimized_policy(job)
# @time println(optimize_prior(pol))

using Glob
files = glob("runs/rando/jobs/*")
jobs = Job.(files)

@sync @distributed for job in jobs
    optimize_prior(job)
end
