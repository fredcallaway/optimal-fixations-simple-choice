using Distributed
addprocs(50)
@everywhere begin
    cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
    include("model.jl")
    include("job.jl")
    include("simulations.jl")
    include("loss.jl")
    CUTOFF = 2000
    using DistributedArrays
end

using Plots
plot([1,2])

# %% ====================  ====================
using Glob
files = glob("runs/rando/jobs/*")
jobs = Job.(files)
policies = optimized_policy.(jobs) |> skipmissing |> collect

@time all_sims = @DArray [simulate_experiment(pol, (µ_emp, σ_emp), 1)
                          for pol in policies];

# all_sims = pmap(jobs) do job
#     prior = (μ_emp, σ_emp)
#     pol = optimized_policy(job)
#     if ismissing(pol)
#         return missing
#     end
#     simulate_experiment(pol, prior, 10)
# end

# %% ====================  ====================
@everywhere include("loss.jl")
@everywhere _loss_funcs = [
    make_loss(value_choice),
    make_loss(fixation_bias),
    make_loss(value_bias),
    make_loss(fourth_rank, :integer),
    make_loss(first_fixation_duration),
    make_loss(last_fixation_duration),
    # make_loss(difference_nfix),
    make_loss(fixation_times, :integer),
    make_loss(last_fix_bias),
    make_loss(gaze_cascade, :integer),
    make_loss(fixate_on_best, Binning(0:CUTOFF/7:CUTOFF)),
    # make_loss(old_value_choice, :integer),
    # make_loss(fixation_value, Binning(0:3970/20:3970)),
]

L = map(all_sims) do sim
    map(_loss_funcs) do loss
        loss(sim)
    end
end
L = combinedims(convert(Array, L))

best = argmin(sum(L; dims=1)[:])  # 65
# sim = simulate_experiment(policies[63],  (μ_emp, σ_emp), 10)
mean(length.(sim.fixations) .> 3)
mean(length.(trials.fixations) .> 3)

# %% ====================  ====================

map(all_sims) do sim
    x = fourth_rank(sim)
    ismissing(x) ? -1. : x[2][1]
end |> argmax
fourth_rank(all_sims[7])

# # %% ====================  ====================
# new_losses = pmap(all_sims) do sim
#     try
#         ismissing(sim) ? Inf : loss(sim)
#     catch
#         return missing
#     end
# end
#
# best = argmin(new_losses)
# sim = all_sims[best]

# current best: 63
# policy = optimized_policy(jobs[best])
# sim = simulate_experiment(policy, (μ_emp, σ_emp), 10)
# fixation_bias, value_choice, value_bias: 91
# fixation_bias value_choice last_fix_bias value_bias fixation_value: 28
sim = all_sims[63]
# %% ====================  ====================
pyplot()
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
const N_BOOT = 1000
using Bootstrap
using Printf
using Plots: px
estimator = mean
ci = 0.95

function ci_err(estimator, y)
    bs = bootstrap(estimator, y, BalancedSampling(N_BOOT))
    c = confint(bs, BasicConfInt(ci))[1]
    abs.(c[2:3] .- c[1])
end

# function plot_human(bins, x, y)
#     vals = bin_by(bins, x, y)
#     bar(mids(bins), estimator.(vals), yerr=ci_err.(estimator, vals),
#        fill=:white, color=:black, label="")
# end

function plot_human(bins, x, y, type=:line)
    vals = bin_by(bins, x, y)
    if type == :line
        plot(mids(bins), estimator.(vals), yerr=ci_err.(estimator, vals),
              grid=:none,
              color=:black,
              label="",)
    elseif type == :discrete
        bar(mids(bins), estimator.(vals), yerr=ci_err.(estimator, vals),
              grid=:none,
              fill=:white,
              color=:black,
              label="",)
  else
      error("Bad plot type : $type")
  end
end

function plot_model!(bins, x, y, type=:line)
    if type == :line
        plot!(mids(bins), estimator.(bin_by(bins, x, y)),
              line=(:red, :dash),
              label="",)
    elseif type == :discrete
        scatter!(mids(bins), estimator.(bin_by(bins, x, y)),
              grid=:none,
              marker=(5, :diamond, :red, stroke(0)),
              label="",)
    else
        error("Bad plot type : $type")
    end
end

function cross!(x, y)
    vline!([x], line=(:grey, 0.7), label="")
    hline!([y], line=(:grey, 0.7), label="")
end

function plot_comparison(feature, sim, bins=nothing, type=:line)
    hx, hy = feature(trials)
    mx, my = feature(sim)
    bins = make_bins(bins, hx)
    plot_human(bins, hx, hy, type)
    plot_model!(bins, mx, my, type)
    # title!(@sprintf "Loss = %.3f" make_loss(feature, bins)(sim))
end

function fig(f, name)
    _fig = f()
    savefig("figs/$name.pdf")
    _fig
end

# %% ====================  ====================
fig("value_choice") do
    plot_comparison(value_choice, sim)
    xlabel!("Relative item value")
    ylabel!("Probability of choice")
end

fig("fixation_bias") do
    plot_comparison(fixation_bias, sim)
    cross!(0, 1/3)
    xlabel!("Relative fixation time")
    ylabel!("Probability of choice")
end

fig("value_bias") do
    plot_comparison(value_bias, sim)
    xlabel!("Relative item value")
    ylabel!("Proportion fixation time")
end

fig("difference_time") do
    plot_comparison(difference_time, sim)
    xlabel!("Maximum relative item value")
    ylabel!("Total fixation time")
end
#
# fig("difference_nfix") do
#     plot_comparison(difference_nfix, sim)
#     xlabel!("Maxium relative item value")
#     ylabel!("Number of fixations")
# end

fig("last_fix_bias") do
    plot_comparison(last_fix_bias, sim)
    cross!(0, 1/3)
    xlabel!("Last fixated item relative value")
    ylabel!("Probability of choosing\nlast fixated item")
end

fig("gaze_cascade") do
    plot_comparison(gaze_cascade, sim, :integer)
    xlabel!("Fixation number (aligned to choice)")
    ylabel!("Proportion of fixations\nto chosen item")
end

fig("fixate_on_best") do
    plot_comparison(fixate_on_best, sim, Binning(0:CUTOFF/7:CUTOFF))
    xlabel!("Time (ms)")
    ylabel!("Probability of fixating\non highest-value item")
end

fig("first_fixation_duration") do
    plot_comparison(first_fixation_duration, sim)
    xlabel!("Duration of first fixation")
    ylabel!("Probability of choice")
end


# argmin(L[4, :])
# sim1 = simulate_experiment(policies[71],  (μ_emp, σ_emp), 10)
fig("fourth_rank") do
    plot_comparison(fourth_rank, sim, :integer, :discrete)
    xlabel!("Value rank of fourth-fixated item")
    ylabel!("Proportion")
    xticks!(1:3, ["best", "middle", "worst"])
end

fig("fixation_times") do
    plot_comparison(fixation_times, sim, :integer, :discrete)
    xticks!(1:4, ["first", "second", "middle", "last"])
    xlabel!("Fixation type")
    ylabel!("Fixation duration")
end

fig("3b_alt") do
    plot_comparison(last_fixation_duration, sim)
    xlabel!("Chosen item time advantage\nbefore last fixation")
    ylabel!("Last fixation duration")
end


# %% ====================  ====================
function fig3b(trials)
    map(trials) do t
        last = t.fixations[end]
        # last != t.choice && return missing
        tft = total_fix_time(t)
        tft[last] -= t.fix_times[end]
        adv = 2 * tft[t.choice] - sum(tft)
        # adv = tft[t.choice] - mean(tft)
        (t.fix_times[end], adv)
    end |> skipmissing |> collect |> invert
end
fig("3b") do
    plot_comparison(fig3b, sim, Binning(-200:400:1400))
    xlabel!("Last fixation duration")
    ylabel!("Chosen item time advantage\nbefore last fixation")
    xticks!(0:400:1200)
end

# %% ====================  ====================


# my_losses = pmap(all_sims) do sim1
#     myloss = make_loss(fixation_times, :integer)
#     ismissing(sim1) ? Inf : myloss(sim1)
# end

# %% ====================  ====================
function neg_fixation_bias(trials)
    mapmany(trials) do t
        ft = total_fix_time(t)
        # invert((ft ./ sum(ft), t.choice .== 1:3))
        invert((ft .- mean(ft), t.choice .== 1:3))[t.value .< 2]
    end |> Vector{Tuple{Float64, Bool}} |> invert
end
plot_comparison(neg_fixation_bias, sim)

# %% ====================  ====================
using StatPlots

function fixate_probs(trials; k=6)
    X = zeros(Int, k, 3)
    for t in trials
        # length(t.fixations) < k && continue
        length(unique(t.value)) != length(t.value) && continue
        ranks = sortperm(-t.value)
        for i in 1:min(k, length(t.fixations))
            r = ranks[t.fixations[i]]
            X[i, r] += 1
        end
    end
    X ./ sum(X, dims=2)
end
Xh = fixate_probs(trials)
Xm = fixate_probs(sim)

fig("4a") do
    groupedbar(Xh, bar_position=:stack, label=["best" "middle" "worst"])
    scatter!(cumsum(Xm; dims=2)[:, 1:2], color=:black, label="")
    xlabel!("Fixation number")
    ylabel!("Proportion of fixations")
end

fig("4a_alt") do
    plot(Xh[:, [1,3]], color=:black, label=["best" "worst"], ls=[:solid :dash])
    plot!(Xm[:, [1,3]], color=:red, label="", ls=[:solid :dash])
    xlabel!("Fixation number")
    ylabel!("Proportion of fixations")
end
# %% ====================  ====================
Xh .- Xm

# %% ====================  ====================
function fixate_probs_from_end(trials; k=6)
    X = zeros(Int, k, 3)
    for t in trials
        # length(t.fixations) < k && continue
        length(unique(t.value)) != length(t.value) && continue
        ranks = sortperm(-t.value)
        nfix = length(t.fixations)
        for i in 1:min(k, nfix)
            r = ranks[t.fixations[end+1-i]]
            X[i, r] += 1
        end
    end
    reverse(X ./ sum(X, dims=2); dims=1)
end
Xh = fixate_probs_from_end(trials)
Xm = fixate_probs_from_end(sim)

fig("4b") do
    groupedbar(Xh, bar_position=:stack, label=["best" "middle" "worst"])
    scatter!(cumsum(Xm; dims=2)[:, 1:2], color=:black, label="")
    xticks!(1:6, string.(-5:0))
    xlabel!("Fixation number")
    ylabel!("Proportion of fixations")
end

fig("4b_alt") do
    plot(Xh[:, [1,3]], color=:black, label=["best" "worst"], ls=[:solid :dash])
    plot!(Xm[:, [1,3]], color=:red, label="", ls=[:solid :dash])
    xticks!(1:6, string.(-5:0))
    xlabel!("Fixation number")
    ylabel!("Proportion of fixations")
end

# %% ====================  ====================
function last_fix_bias(trials)
    map(trials) do t
        last = t.fixations[end]
        if t.value[last] > μ_emp
            return missing
        end
        return (t.value[last] - mean(t.value), t.choice == last)
    end |> skipmissing |> collect |> invert
end

plot_comparison(last_fix_bias, sim)
cross!(0, 1/3)
xlabel!("Last fixated item relative value")
ylabel!("Probability of choosing\nlast fixated item")

# %% ====================  ====================

function bar(t)::Vector{Float64}
    x = zeros(3)
    nfix = length(t.fixations)
    mid = cld(nfix, 2)
    for i in mid:nfix
        x[t.fixations[i]] += t.fix_times[i]
    end
    return x
end

function foo(trials)
    mapmany(trials) do t
        tft = bar(t)
        invert((t.value .- mean(t.value), tft ./ sum(tft)))
    end |> invert
end

# %% ====================  ====================

function best_vs_middle(trials)
    # x, y = [], []
    map(trials) do t
        ft = total_fix_time(t)
        best, middle, worst = sortperm(-t.value)
        t.value[best] == t.value[middle] && return missing
        t.choice == worst && return missing
        ft[best] - ft[middle], t.choice == best
    end |> skipmissing |> collect |> invert
end

fig("best_vs_middle") do
    plot_comparison(best_vs_middle, sim)
    xlabel!("Fixation time on best vs. middle")
    ylabel!("Probability choose best")
end

# %% ====================  ====================

function not_worst(trials)
    # x, y = [], []
    map(trials) do t
        ft = total_fix_time(t)
        best, middle, worst = sortperm(-t.value)
        t.value[middle] == t.value[worst] && return missing
        # t.choice == worst && return missing
        ft[best] + ft[middle] - ft[worst], t.choice != worst
    end |> skipmissing |> collect |> invert
end

fig("not_worst") do
    plot_comparison(not_worst, sim)
    xlabel!("Fixation time on best or middle vs. worst")
    ylabel!("Probability choose best or middle  ")
end

# # %% ====================  ====================
# @everywhere function fixate_on_best(trials; k=6)
#     denom = zeros(Int, k)
#     num = zeros(Int, k)
#     for t in trials
#         x = t.fixations .== argmin(t.value)
#         for i in 1:min(k, length(x))
#             denom[i] += 1
#             num[i] += x[i]
#         end
#     end
#     1:k, num ./ denom
# end
# plot_comparison(fixate_on_best, sim, :integer)

# %% ====================  ====================
@everywhere myloss = make_loss(fixate_on_best, Binning(0:CUTOFF/7:CUTOFF))
my_losses = pmap(all_sims) do sim1
    ismissing(sim1) ? Inf : myloss(sim1)
end
my_best = argmin(my_losses)
sim1 = all_sims[my_best]
plot_comparison(fixate_on_best, sim1, Binning(0:CUTOFF/7:CUTOFF))

# plot_comparison(fixation_value, sim, Binning(0:3970/20:3970))
# %% ====================  ====================

# myloss = make_loss(fixation_value, Binning(0:3970/20:3970))
# my_losses = pmap(all_sims) do sim1
#     ismissing(sim1) ? Inf : myloss(sim1)
# end
# my_best = argmin(my_losses)
# sim1 = all_sims[my_best]

# plot_comparison(fixation_value, sim, Binning(0:3970/20:3970))
# fig("fixation_series") do
#     plot_comparison(fixation_value, sim, Binning(0:3970/20:3970))
#     xlabel!("Time (ms)")
#     ylabel!("Ave")
# end

# %% ====================  ====================



# %% ====================  ====================
function wow(trials)
    x, y, z = [], [], []
    for t in trials
        tft = total_fix_time(t)
        rv = t.value .- mean(t.value)
        c = t.choice .== 1:3
        for i in 1:3
            push!(x, tft[i]))
            push!(y, rv[i]))
            push!(z, c[i]))
    end
end
# %% ====================  ====================
function bang(trials)
    map(trials) do t
        tft = total_fix_time(t)
        rv = t.value .- mean(t.value)
        c = t.choice
        tft[c], rv[c]
    end |> invert
end

bang(trials)

# %% ====================  ====================



# %% ====================  ====================
# priors, losses = map(jobs) do job
#     try
#         deserialize(job, :optimized_prior)
#     catch
#         (missing, Inf)
#     end
# end |> invert
#
# best = argmin(new_losses)
# job = jobs[best]
# policy = optimized_policy(job)
# prior = priors[best]
# sim = simulate_experiment(policy, prior)
# # @time sim = simulate_experiment(optimized_policy(job), prior, 10)
# %% ====================  ====================