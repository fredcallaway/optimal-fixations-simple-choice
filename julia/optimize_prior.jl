using Distributed
using SplitApplyCombine
# cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
@everywhere begin
    include("simulations.jl")
    include("loss.jl")

    function optimize_prior(job)
        pol = optimized_policy(job)
        if ismissing(pol)
            return
        end

        μs = [0:0.1:μ_emp; μ_emp]
        ismissing(pol) && return missing
        @time x = map(μs) do μ
            # (prior=(μ, σ_emp), sim=nothing, loss=rand())
            sim = simulate_experiment(pol, (µ, σ_emp))
            (prior=(μ, σ_emp), sim=sim, losses=breakdown_loss(sim))
        end
        serialize(job, :simulations_alt, x)
    end

end # everywhere

# %% ====================  ====================

using Glob
files = glob("runs/rando/jobs/*")
jobs = Job.(files)

@sync @distributed for job in jobs
    optimize_prior(job)
end
