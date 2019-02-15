pwd()
pop!(ARGS)
cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
# %% ====================  ====================
include("model.jl")
include("human.jl")
include("job.jl")
include("simulations.jl")
include("binning.jl")
# %% ====================  ====================
# D = mapmany(trials) do t
#     d = discretize_fixations(t)
#     ranks = sortperm(sortperm(-t.value))
#     enumerate(ranks[d])
# end |> invert
# const human_fix_ranks = Table(time=D[1] * 50, focus=D[2])
# CSV.write("../krajbich_PNAS_2011/discretized_fixations.csv", human_fix_ranks)
# CSV.write("../krajbich_PNAS_2011/trials.csv", trials)

# %% ====================  ====================

using Plots
using Glob
files = glob("runs/rando/jobs/*")
jobs = Job.(files)
pol = optimized_policy(jobs[44])
sim = simulate_experiment(pol)
time_per_sample = mean(sum.(trials.fix_times)) / mean(length.(sim.fixations))

# %% ====================  ====================
using Bootstrap
estimator = mean
ci = 0.95

function ci_err(estimator, y)
    bs = bootstrap(estimator, y, BalancedSampling(1000))
    c = confint(bs, BasicConfInt(ci))[1]
    abs.(c[2:3] .- c[1])
end

function plot_human(bins, x, y)
    vals = bin_by(bins, x, y)
    bar(mids(bins), estimator.(vals), yerr=ci_err.(estimator, vals),
       fill=:white, color=:black, label="")
end

function plot_model!(bins, x, y)
    plot!(mids(bins), estimator.(bin_by(bins, x, y)),
          shape=:circle, markerstrokecolor=:red, color=:red, markercolor=:white,
          label="")
end

# %% ==================== fixation time -> choice ====================
function total_fix_time(t)
    x = zeros(3)
    for (fi, ti) in zip(t.fixations, t.fix_times)
        x[fi] += ti
    end
    x
end

function fixation_bias_human()
    fix_diff = map(trials) do t
        x = total_fix_time(t)
        x[1] - mean(x)
    end
    fix_diff, trials.choice .== 1
end

function fixation_bias_model()
    fix_diff = map(sim.fixations) do f
        c = counts(f, 3)
        (c[1] - mean(c)) * time_per_sample
    end
    fix_diff, sim.choice .== 1
end

bins = Binning(-700:200:700)
plot_human(bins, fixation_bias_human()...)
plot_model!(bins, fixation_bias_model()...)


# %% ==================== value difference -> time ====================
function rt_val_std_loss(sim)
    n_sim = Int(length(sim) / length(trials))
    n_sample = length.(sim.fixations)
    time_per_sample = mean(fix_time) / mean(n_sample)
    pred_fix_time = n_sample * time_per_sample
    mpe(bin_means(val_std, fix_time), bin_means(repeat(val_std, n_sim), pred_fix_time))
end

difficulty(v) = maximum(v) - mean(v)

bins = Binning(1:6)
plot_human(bins, difficulty.(trials.value), sum.(trials.fix_times))
plot_model!(bins, difficulty.(sim.value), length.(sim.fixations) .* time_per_sample)

# %% ==================== Figure 3e ====================

hx, hy = map(trials) do t
    t.fix_times[1], t.choice == t.fixations[1]
end |> invert

function fixations(s)
    fixations = Int[]
    fix_times = Float64[]
    prev = nothing
    for f in s.fixations
        if f != prev
            prev = f
            push!(fixations, f)
            push!(fix_times, 0.)
        end
        fix_times[end] += time_per_sample
    end
    fixations, fix_times
end

mx, my = map(sim) do s
    fixs, fix_times = fixations(s)
    fix_times[1], s.choice == fixs[1]
end |> invert

bins = Binning(100:200:900)
plot_human(bins, hx, hy)
plot_model!(bins, mx, my)


# %% ====================  ====================
using DataFrames
d = DataFrame(fix_diff=fix_diff, p_choose=Int.(p_choose))
R"""
library(ggplot2)
ggplot($d, aes(x=fix_diff, y=p_choose)) +
  stat_smooth(method="glm", method.args=list(family="binomial"), se=FALSE)
"""
# %% ====================  ====================
R"""
data = $d
data$bins <- cut(data$fix_diff, breaks = 10)
# Points:
ggplot(data, aes(x = bins, y = p_choose)) +
  stat_summary(fun.y = "mean", geom = "point")

# Histogram bars:
# ggplot(data, aes(x = bins, y = y)) +
# stat_summary(fun.y = "mean", geom = "histogram")
"""
