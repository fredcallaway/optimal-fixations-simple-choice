# pop!(ARGS)
cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
using Distributed
addprocs(44)
@everywhere include("simulations.jl")
@everywhere include("loss.jl")

using Glob
files = glob("runs/rando/jobs/*")
jobs = Job.(files)

# # %% ====================  ====================
# function optimize_prior(pol)
#     candidates = reverse(0:0.1:round(μ_emp; digits=1))
#     losses = pmap(candidates) do μ
#         loss(simulate_experiment(pol, (μ, σ_emp)))
#     end
#     best = argmin(losses)
#     (candidates[best], σ_emp), losses[best]
# end
#
function get_prior(how::Symbol, pol)
    how == :optimize ? optimize_prior(pol) :
    how == :empirical ? (μ_emp, σ_emp) :
    how == :zero ? (0, σ_emp) :
    error("$how is not a valid argument for `how`")
end
# get_prior(prior::Tuple{Real, Real}, pol) = prior

function run_simulation(job, prior)
    pol = optimized_policy(job)
    ismissing(pol) && return (missing, missing, Inf)
    prior = get_prior(prior, pol)
    sim = simulate_experiment(pol, prior)
    serialize(job, Symbol("simulation_$prior"), sim)
    # file = save(job, Symbol("simulation_$prior"), invert(sim))
    (sim, prior, loss(sim))
end



# @everywhere include("job.jl")
# job = jobs[1]
# deserialize(job, :test)
# # %% ====================  ====================
# @time opt_results = map(jobs) do job
#     run_simulation(job, :optimize)
#     print("x")
# end
# opt_results
#
# @time emp_results = pmap(jobs) do job
#     run_simulation(job, :empirical)
# end
# # %% ====================  ====================
# sims, priors, losses, files = invert(emp_results)
# losses[ismissing.(losses)] .= Inf
# losses = Vector{Float64}(losses)
# idx = argmin(losses)
# pol = optimized_policy(jobs[idx])
# sim = run_simulation(jobs[idx], (3.8, σ_emp))[1]
# # mkdir("sim_results")
# open("sim_results/empirical", "w+") do f
#     serialize(f, sim)
# end
#
# @time sim = run_simulation(jobs[1], :empirical);
# @time loss(sim);
#
#
# # %% ==================== Scratch ====================
# pred = get_fix_ranks(sim)
#
# rt_cutoff = quantile(human_fix_ranks.time, 0.95)
# binning = Binning(collect(50:50:rt_cutoff))
# bins = binning.(pred.time)
#
#
# group(x->x[1], x->x[2], zip(bins, y)) |>
#   sort |> values |> collect .|> mean
# end
#
#
#
# n_sample = length.(sim.fixations)
# time_per_sample = mean(fix_time) / mean(n_sample)
# pred = get_fix_ranks(sims[best])
# pred_time = pred.step * time_per_sample
#
#
# hcat([bin_means(pred.step, pred.focus .== i; n=10) for i in (1,3)]...)
#
# # %% ====================  ====================
# # X = (multi_loss.(skipmissing(sims)))
# # multi_loss = juxt(rt_val_std_loss, mean_n_fix_loss, choice_val_loss)
#
# losses = loss.(sims)
# best = argmin(losses)
# sim_files[best]
#
# fix_rank_loss(sims[best])
#
# fix_rank_target
#
# length.(sims[best].fixations)
# jobs[best]
#
# jobs[best]
#
# idx = findall(.!ismissing.(sims))
#
# findall(ismissing.(sims))
# ratio_cutoff = Dict(i=>mean(length.(sims[i].fixations) .== 99) for i in idx)
#
# countmap([jobs[i].sample_cost for i in idx])
# countmap([j.sample_cost for j in jobs])
#
# group(i->jobs[i].sample_cost, i->ratio_cutoff[i], idx) |>
#   sort |> values .|> mean
#
# # %% ====================  ====================
