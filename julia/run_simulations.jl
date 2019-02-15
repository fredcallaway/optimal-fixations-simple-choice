# pop!(ARGS)
cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
using Distributed
# addprocs(40)
@everywhere include("simulations.jl")

using Glob
files = glob("runs/rando/jobs/*")
jobs = Job.(files)


# %% ====================  ====================
PARALLEL_SIM = false

@everywhere function run_simulation(job, prior)
    pol = optimized_policy(job)
    ismissing(pol) && return (missing, missing, missing, missing)
    μ, σ = prior == :optimize ? optimize_prior(pol; max_func_evals=100) :
           prior == :empirical ? (μ_emp, σ_emp) :
           prior == :zero ? (0, σ_emp) :
           error("$prior is not a valid prior argument")

    sim = simulate_experiment(pol; μ=μ, σ=σ)
    file = save(job, Symbol("simulation_$prior"), invert(sim))
    (sim, (μ, σ), loss(sim), file)
end

# %% ====================  ====================
@time opt_results = pmap(jobs) do job
    run_simulation(job, :optimize)
end

@time emp_results = pmap(jobs) do job
    run_simulation(job, :empirical)
end

# %% ==================== Scratch ====================
pred = get_fix_ranks(sim)

rt_cutoff = quantile(human_fix_ranks.time, 0.95)
binning = Binning(collect(50:50:rt_cutoff))
bins = binning.(pred.time)


group(x->x[1], x->x[2], zip(bins, y)) |>
  sort |> values |> collect .|> mean
end



n_sample = length.(sim.fixations)
time_per_sample = mean(fix_time) / mean(n_sample)
pred = get_fix_ranks(sims[best])
pred_time = pred.step * time_per_sample


hcat([bin_means(pred.step, pred.focus .== i; n=10) for i in (1,3)]...)

# %% ====================  ====================
# X = (multi_loss.(skipmissing(sims)))
# multi_loss = juxt(rt_val_std_loss, mean_n_fix_loss, choice_val_loss)

losses = loss.(sims)
best = argmin(losses)
sim_files[best]

fix_rank_loss(sims[best])

fix_rank_target

length.(sims[best].fixations)
jobs[best]

jobs[best]

idx = findall(.!ismissing.(sims))

findall(ismissing.(sims))
ratio_cutoff = Dict(i=>mean(length.(sims[i].fixations) .== 99) for i in idx)

countmap([jobs[i].sample_cost for i in idx])
countmap([j.sample_cost for j in jobs])

group(i->jobs[i].sample_cost, i->ratio_cutoff[i], idx) |>
  sort |> values .|> mean

# %% ====================  ====================
