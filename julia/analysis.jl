pwd()
pop!(ARGS)
cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
# %% ====================  ====================
include("model.jl")
include("human.jl")
include("job.jl")
include("simulations.jl")
include("binning.jl")


# %% ==================== Explore data ====================

num_fixation(trials[1].fixations)
trials[2].fixations

map(typeof, group(x->x.subject, trials))

@>> begin
    trials
    group(x->x.subject)
    values
    map
end


groupsum(x->x.subject, x->length(x.fixations), trials)
groupsum(x->x.subject, x->x.rt, trials)
groupsum(x->x.subject, x->x.choice == argmax(x.value), trials)

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

sim = simulate_experiment(pol; n_repeat=10, μ=4.)
# %% ====================  ====================
using Serialization

sim = open("sim_results/empirical") do f
    deserialize(f)
end
# %% ====================  ====================

function mpe(y, yhat)
    @assert size(y) == size(yhat)
    mean(abs.(y .- yhat) ./ y)
end

function make_bins(bins, hx)
    if bins == :integer
        return Binning(minimum(hx)-0.5:1:maximum(hx)+0.5)
    elseif bins isa Nothing
        bins = 5
    end
    if bins isa Int
        low, high = quantile(hx, [0.05, 0.95])
        bin_size = (high - low) / bins
        bins = Binning(low:bin_size:high)
        # bins = Binning(quantile(hx, 0:1/bins:1))
    end
    return bins
end

function make_loss(feature::Function, bins=nothing)
    hx, hy = feature(trials)
    bins = make_bins(bins, hx)
    h = bin_by(bins, hx, hy) .|> mean
    (sim) -> begin
         m = bin_by(bins, feature(sim)...) .|> mean
         err = mpe(h, m)
         isnan(err) ? Inf : err
    end
end


# %% ====================  ====================

const N_BOOT = 10
using Bootstrap
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


function plot_human(bins, x, y)
    vals = bin_by(bins, x, y)
    plot(mids(bins), estimator.(vals), yerr=ci_err.(estimator, vals),
          color=:black,
          label="")
end

function plot_model!(bins, x, y)
    plot!(mids(bins), estimator.(bin_by(bins, x, y)),
          line=(:red, :dash),
          label="")
end

function plot_comparison(feature, sim, bins=nothing)
    hx, hy = feature(trials)
    mx, my = feature(sim)
    bins = make_bins(bins, hx)
    plot_human(bins, hx, hy)
    plot_model!(bins, mx, my)
end


# %% ==================== fixation time -> choice ====================

function total_fix_time(t)::Vector{Float64}
    x = zeros(3)
    for (fi, ti) in zip(t.fixations, t.fix_times)
        x[fi] += ti
    end
    return x
end

function fixation_bias(trials)
    mapmany(trials) do t
        ft = total_fix_time(t)
        # invert((ft ./ sum(ft), t.choice .== 1:3))
        invert((ft .- mean(ft), t.choice .== 1:3))
    end |> Vector{Tuple{Float64, Bool}} |> invert
end

plot_comparison(fixation_bias, sim, 8)
xlabel!("Fixation advantage")
ylabel!("Probability of choice")


# %% ==================== value difference -> time ====================

difficulty(v) = maximum(v) - mean(v)
# hx = difficulty.(trials.value)
# quantile(hx, [0.05, 0.95])

function difference_time(trials)
    difficulty.(trials.value), sum.(trials.fix_times)
end

plot_comparison(difference_time, sim, 5)
xlabel!("Max value minus mean value")
ylabel!("Total fixation time")


# %% ==================== max value -> choice value ====================

choice_value(t) = t.value[t.choice]

function value_choice(trials)
    Int.(maximum.(trials.value)), choice_value.(trials)
end

plot_comparison(value_choice, sim, :integer)

# %% ==================== First fixation duration -> choose first fixated ====================


function first_fixation_bias(trials)
    map(trials) do t
        t.fix_times[1], t.choice == t.fixations[1]
    end |> invert
end

plot_comparison(first_fixation, sim, 5)


# %% ==================== Last fixation chosen ====================

choose_last_fixated(t) = t.fixations[end] == t.choice

function last_fix_bias(trials)
    map(trials) do t
        last = t.fixations[end]
        t.value[last] - mean(t.value), t.choice == last
    end |> invert
end
# x, y = last_fix_bias(trials)
# quantile(x, [0.05, 0.95])

plot_comparison(last_fix_bias, sim, 8)
hline!([1/3], line=(:black, :dot), label="")
vline!([0], line=(:black, :dot), label="")
xlabel!("Relative value of last fixated item")
ylabel!("Probability of choosing last fixated item")


# %% ==================== value -> fixation time ====================

function value_bias(trials)
    mapmany(trials) do t
        tft = total_fix_time(t)
        invert((t.value .- mean(t.value), tft ./ sum(tft)))
    end |> invert
end

plot_comparison(value_bias, sim)
xlabel!("Relative value")
ylabel!("Proportion fixation time")


losses = [
    make_loss(fixation_bias),
    make_loss(fixation_bias),
    make_loss(difference_time),
    make_loss(value_choice, :integer),
    make_loss(first_fixation_bias),
    make_loss(last_fix_bias),
    make_loss(value_bias),
]

function loss(sim)
    sum(l(sim) for l in losses)
end
