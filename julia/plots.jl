
using Distributed
nprocs() == 1 && addprocs()
@everywhere begin
    cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
    include("model_base.jl")
    include("dc.jl")
end
include("features.jl")
using Serialization
using Plots
plot([1,2])

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
    return sem(y) * 2
    bs = bootstrap(estimator, y, BalancedSampling(N_BOOT))
    c = confint(bs, BasicConfInt(ci))[1]
    abs.(c[2:3] .- c[1])
end

function plot_human!(bins, x, y, type=:line; kws...)
    vals = bin_by(bins, x, y)
    if type == :line
        plot!(mids(bins), estimator.(vals), yerr=ci_err.(estimator, vals),
              grid=:none,
              line=(2,),
              color=:black,
              label="";
              kws...)
    elseif type == :discrete
        Plots.bar!(mids(bins), estimator.(vals),
            yerr=ci_err.(estimator, vals),
              grid=:none,
              fill=:white,
              line=(2,),
              color=:black,
              label="";
              kws...)
  else
      error("Bad plot type : $type")
  end
end


function plot_model!(bins, x, y, type=:line; kws...)
    vals = bin_by(bins, x, y)
    if type == :line
        plot!(mids(bins), estimator.(vals),
              # yerr=ci_err.(estimator, vals),
              color=RED,
              line=(RED, :dash, 2),
              label="";
              kws...)
    elseif type == :discrete
        scatter!(mids(bins), estimator.(vals),
              # yerr=ci_err.(estimator, vals),
              grid=:none,
              marker=(5, :diamond, RED, stroke(0)),
              label="";
              kws...)
    else
        error("Bad plot type : $type")
    end
end

function cross!(x, y)
    vline!([x], line=(:grey, 0.7), label="")
    hline!([y], line=(:grey, 0.7), label="")
end

function plot_comparison(feature, sim, bins=nothing, type=:line; kws...)
    hx, hy = feature(trials; kws...)
    mx, my = feature(sim; kws...)
    bins = make_bins(bins, hx)
    plot()
    plot_human!(bins, hx, hy, type)
    plot_model!(bins, mx, my, type)
    # title!(@sprintf "Loss = %.3f" make_loss(feature, bins)(sim))
end


function fig(f, name)
    _fig = f()
    savefig("figs/$run_name/$name.pdf")
    _fig
end

using KernelDensity

# %% ====================  ====================
function kdeplot!(k::UnivariateKDE, xmin, xmax; kws...)
    plot!(range(xmin, xmax, length=200), z->pdf(k, z); grid=:none, label="", kws...)
end

function kdeplot!(x; xmin=quantile(x, 0.05), xmax=quantile(x, 0.95), kws...)
    kdeplot!(kde(x), xmin, xmax; kws...)
end

function kdeplot!(x, bw::Float64; xmin=quantile(x, 0.05), xmax=quantile(x, 0.95), kws...)
    kdeplot!(kde(x, bandwidth=bw), xmin, xmax; kws...)
end

# %% ==================== Load Blinkered ====================
using Serialization
run_name = "1000"
mkpath("figs/$run_name")
results = Dict(
    "fit5" => "results/2019-06-02T11-49-47",
    "fit6" => "results/2019-06-01T11-36-33",
    "fit4" => "results/2019-06-13T15-01-18",
    "fit_new" => "results/2019-06-13T17-29-40",
    "intensive" => "results/2019-06-14T13-09-01",
    "reweight" => "results/2019-06-14T17-39-42",
    "fit4-2000" => "results/2019-06-15T13-34-26/",
    "fit3" => "results/2019-06-16T19-01-26/",
    "fit5-alpha-100" => "results/2019-06-17T15-50-40/",
    "fit5-alpha-100-reweight" => "results/2019-06-17T21-52-40/",
    "1000" => "results/inference/2019-06-21T13-57-39"
)[run_name]

try
    policy, prior = open(deserialize, "$results/mle")
catch
    policy, prior = open(deserialize, "$results/blinkered_policy.jls")
end
@time sim = simulate_experiment(policy, prior, 100,
    sample_time=sample_time, parallel=true)

# %% ====================  ====================
run_name = "moments/3/gp_min"
mkpath("figs/$run_name")

# policy, prior, sample_time = open(deserialize, "results/moments/3/gp_min/2019-06-21T23-38-38/best")
# policy, prior, sample_time = open(deserialize, "results/moments/3/gp_min/2019-07-02T10-23-35-D4x/best")

#
# include("results.jl")
# result = get_results(run_name)[5]
# policy, prior, sample_time = load(result, :best)
# @time sim = simulate_experiment(policy, prior, 100, sample_time=sample_time, parallel=true)

# %% ====================  ====================
run_name = "additive_bmps"
mkpath("figs/$run_name")
sim = open(deserialize, "tmp/best_bmps_sim")

# %% ====================  ====================
# TODO: total value -> n fixation

fig("value_choice") do
    plot_comparison(value_choice, sim, :integer)
    cross!(0, 1/3)
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
    cross!(0, 1/3)
    xlabel!("Relative item value")
    ylabel!("Proportion fixation time")
end

fig("fixate_on_best") do
    # FIXME Incorrect error bars!
    plot_comparison(fixate_on_best, sim, :integer, cutoff=2000, n_bin=8)
    xticks!(0.5:8.5, string.(0:250:2000))

    # plot_comparison(fixate_on_best, sim, :integer, cutoff=2000, n_bin=5)
    # xticks!(0.5:5.5, string.(0:400:2000))

    hline!([1/3], line=(:grey, 0.7), label="")
    xlabel!("Time since trial onset")
    ylabel!("Probability of fixating\non highest-value item")
end

fig("fourth_rank") do
    plot_comparison(fourth_rank, sim, :integer, :discrete)
    xlabel!("Value rank of fourth-fixated item")
    ylabel!("Proportion")
    xticks!(1:3, ["best", "middle", "worst"])
end

fig("first_fixation_duration") do
    plot_comparison(first_fixation_duration, sim, )
    xlabel!("Duration of first fixation")
    ylabel!("Probability choose first fixated")
end

fig("last_fixation_duration") do
    plot_comparison(last_fixation_duration, sim)
    xlabel!("Chosen item time advantage\nbefore last fixation")
    ylabel!("Last fixation duration")
end

fig("difference_time") do
    plot_comparison(difference_time, sim)
    xlabel!("Maximum relative item value")
    ylabel!("Total fixation time")
end

fig("difference_nfix") do
    plot_comparison(difference_nfix, sim, :integer)
    xlabel!("Maxium relative item value")
    ylabel!("Number of fixations")
end

fig("fixation_times") do
    plot_comparison(fixation_times, sim, :integer, :discrete)
    xticks!(1:4, ["first", "second", "middle", "last"])
    xlabel!("Fixation type")
    ylabel!("Fixation duration")
end

fig("n_fix_hist") do
    plot_comparison(n_fix_hist, sim, :integer, :discrete)
    xlabel!("Number of fixations")
    ylabel!("Proportion of trials")
end

fig("last_fix_bias") do
    plot_comparison(last_fix_bias, sim, :integer)
    cross!(0, 1/3)
    xlabel!("Last fixated item relative value")
    ylabel!("Probability of choosing\nlast fixated item")
end

fig("gaze_cascade") do
    plot_comparison(gaze_cascade, sim, :integer)
    xlabel!("Fixation number (aligned to choice)")
    ylabel!("Proportion of fixations\nto chosen item")
end

fig("rt_kde") do
    plot(xlabel="Total fixation time", ylabel="Probability density")
    kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=5000, line=(:black, 2), )
    kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=5000, line=(RED, :dash, 2), )
end

fig("chosen_fix_time") do
    plot_comparison(chosen_fix_time, sim, :integer, :discrete)
    xticks!(0:1, ["Unchosen", "Chosen"])
    ylabel!("Average fixation duration")
end

fig("value_duration") do
    plot_comparison(value_duration, sim, :integer)
    xlabel!("Item value")
    ylabel!("Fixation duration")
end


fig("value_duration_first") do
    plot_comparison(value_duration_first, sim, :integer)
    xlabel!("First fixated item value")
    ylabel!("Fixation duration")
end
# fig("value_duration") do
#     plot_comparison(value_duration, sim, :integer)
#     xlabel!("Item value")
#     ylabel!("Average fixation duration")
# end

# %% ====================  ====================

function plot_human!(feature::Function, bins=nothing, type=:line; kws...)
    hx, hy = feature(trials)
    bins = make_bins(bins, hx)
    plot_human!(bins, hx, hy, type; kws...)
end

function plot_model!(feature::Function, bins=nothing, type=:line; kws...)
    hx, hy = feature(sim)
    bins = make_bins(bins, hx)
    plot_human!(bins, hx, hy, type; kws...)
end

# %% ====================  ====================

policy

# %% ====================  ====================

function value_n_fix(trials)
    x = Float64[]
    y = Int[]
    for t in trials
        push!(x, relative_value(t)...)
        push!(y, counts(t.fixations, 3)...)
    end
    x, y
end

plot_comparison(value_n_fix, sim, :integer)

# %% ====================  ====================

function value_fixtime(trials)
    x = Float64[]
    y = Int[]
    for t in trials
        # rv = ranks = sortperm(sortperm(-t.value))
        push!(x, relative_value(t)...)
        push!(y, total_fix_time(t)...)
    end
    x, y
end

plot_comparison(value_fixtime, sim)

# %% ====================  ====================

function uncertainty_bonus(trials)
    options = Set([1,2,3])
    x = Float64[]
    y = Bool[]
    for t in trials
        cft = zeros(3)
        total = 0
        for i in eachindex(t.fixations)
            fix = t.fixations[i]
            fix_time = t.fix_times[i]
            if i > 1
                prev = t.fixations[i-1]
                alt = pop!(setdiff(options, [prev, fix]))
                d = (cft[fix] - cft[alt]) / total
                if d < 0
                    push!(x, -d)
                    push!(y, true)
                else
                    push!(x, d)
                    push!(y, false)
                end

                # z = cft ./ total
                # z = cft .- total/3
                # push!(x, z[alt])
                # push!(x, z[fix])
                # push!(y, false)
                # push!(y, true)
            end
            cft[fix] += fix_time
            total += fix_time
        end
    end
    return x, y
end
plot_comparison(uncertainty_bonus, sim)
# fig("fixate_uncertain") do
#     xlabel!("Fixat")
# end
# plot()
# fig("uncertainty1") do
#     plot_comparison(uncertainty_bonus, sim)
#     xlabel!("Proportion of previous fixation time")
#     ylabel!("Probability of fixating")
# end
# %% ====================  ====================
function diff_fix(trials)
    options = Set([1,2,3])
    x = Float64[]
    for t in trials
        cft = zeros(3)
        total = 0
        for i in eachindex(t.fixations)
            fix = t.fixations[i]
            fix_time = t.fix_times[i]
            if i > 2
                prev = t.fixations[i-1]
                alt = pop!(setdiff(options, [prev, fix]))
                push!(x, cft[fix] - cft[alt])
            end
            cft[fix] += fix_time
            total += fix_time
        end
    end
    return x
end

fig("refixate_uncertain") do
    plot(xlabel="Fixation advantage of refixated item",
        ylabel="Probability density")
    kdeplot!(diff_fix(trials), 100., line=(:black, 2))
    kdeplot!(diff_fix(sim), 100., line=(:red, 2))
    vline!([0], line=(:grey, 0.7), label="")
end
# plot()
# kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=5000, line=(:black, 2), )
# %% ====================  ====================
t
function diff_fix_fourth(trials)
    options = Set([1,2,3])
    x = Float64[]
    for t in trials
        if length(t.fixations) > 3 && sort(t.fixations[1:3]) == 1:3 && unique_values(t)
            cft = t.fix_times[t.fixations[1:3]]
        end
    end
    return x
end


# %% ====================  ====================
function last_two(trials)
    x = Tuple{Int, Float64}[]
    for t in trials
        f = t.fix_times
        length(f) < 2 && continue
        for i in 0:1
            push!(x, (i, f[end-i]))
        end
    end
    invert(x)
end

plot_comparison(last_two, sim, :integer, :discrete)

# %% ====================  ====================

function fixation_times(trials, n)
    x = Tuple{Int, Float64}[]
    for t in trials
        f = t.fix_times
        length(f) != n && continue
        for i in 1:n
            push!(x, (i, f[i]))
        end
    end
    invert(x)
end

fig("split_fixations") do
    plot()
    c = colormap("Blues", 8)
    for n in 2:8
        plot_human!(x->fixation_times(x, n), :integer, :line, color=c[n])
    end
    xlabel!("Fixation number")
    ylabel!("Fixation duration")
end

fig("split_fixations_model") do
    plot()
    c = colormap("Reds", 8)
    for n in 2:8
        plot_model!(x->fixation_times(x, n), :integer, :line, color=c[n])
    end
    xlabel!("Fixation number")
    ylabel!("Fixation duration")
end

# %% ====================  ====================
function rev_fixation_times(trials)
    x = Tuple{Int, Float64}[]
    for t in trials
        f = t.fix_times
        length(f) < 4 && continue
        for i in 0:min(4, length(f)-1)
            push!(x, (-i, f[end-i]))
        end
    end
    invert(x)
end

function rev_fixation_times(trials, n)
    x = Tuple{Int, Float64}[]
    for t in trials
        f = t.fix_times
        length(f) != n && continue
        for i in 0:n-1
            push!(x, (-i, f[end-i]))
        end
    end
    invert(x)
end

plot()
plot_human!(rev_fixation_times, :integer)
# %% ====================  ====================

fig("split_fixations") do
    plot()
    c = colormap("Blues", 8)
    for n in 2:8
        plot_human!(x->rev_fixation_times(x, n), :integer, :line, color=c[n])
    end
    xlabel!("Fixation number (from final)")
    ylabel!("Fixation duration")
end
# %% ====================  ====================

fig("split_fixations_model") do
    plot()
    c = colormap("Reds", 8)
    for n in 2:8
        plot_model!(x->rev_fixation_times(x, n), :integer, :line, color=c[n])
    end
    xlabel!("Fixation number (from final)")
    ylabel!("Fixation duration")
end

# %% ====================  ====================
fig("fixations_from_end") do
    plot()
    plot_human!(rev_fixation_times, :integer)
    xlabel!("Fixation number (from final)")
    ylabel!("Fixation duration")
end
# %% ====================  ====================
using RCall
function choice_glm(trials)
    rv = Float64[]
    rft = Float64[]
    choice = Bool[]
    for t in trials
        ft = total_fix_time(t)
        push!(rv, (t.value .- mean(t.value))...)
        push!(rft, (ft .- mean(ft))...)
        push!(choice, (1:3 .== t.choice)...)
    end
    R"""
    summary(glm($choice ~ $rv + $rft, family="binomial"))
    """
end
display("")
println("------ Human ------")
println(choice_glm(trials))
println("------ Model ------")
println(choice_glm(sim))
# %% ====================  ====================

function all_ranks(trials)
    early, late = Int[], Int[]
    for t in trials
        if length(t.fixations) > 3
            ranks = sortperm(sortperm(-t.value))
            for i in eachindex(t.fixations)
                x = i < 4 ? early : late
                push!(x, ranks[t.fixations[i]])
            end
        end
    end
    # n = length(x)
    counts(early, 3) ./ length(early), counts(late, 3) ./ length(late)
    # std_ = @. √(p * (1 - p) / n)
    # 1:3, p, std_
end

# %% ====================  ====================

# %% ====================  ====================

function find_switch(fixations)
    seen = Set()
    for i in eachindex(fixations)
        push!(seen, fixations[i])
        length(seen) == 3 && return i
    end
    return 1000
end

function early_late_proportion(trials)
    x = Int[]
    y = Float64[]
    for t in trials
        switch = find_switch(t.fixations)
        length(t.fixations) > switch || continue
        # length(unique(t.fixations[1:3])) == 3 || continue
        ranks = sortperm(-t.value)
        time_on_best = float(ranks[t.fixations] .== 1) .* t.fix_times
        push!(x, 1, 2)
        push!(y, sum(time_on_best[1:switch]) / sum(t.fix_times[1:switch]))
        push!(y, sum(time_on_best[switch+1:end]) / sum(t.fix_times[switch+1:end]))
    end
    x, y
end

plot_comparison(early_late_proportion, sim, :integer, :discrete)
hline!([1/3], line=(:grey, 0.7), label="")
xticks!(1:2, ["Early", "Late"])
ylabel!("Proportion fixation time on best")


# %% ====================  ====================
function dwell_time(trials)
    chosen, other = Float64[], Float64[]
    for t in trials
        for i in eachindex(t.fixations)
            if t.fixations[i] == t.choice
                push!(chosen, t.fix_times[i])
            else
                push!(other, t.fix_times[i])
            end
        end
    end
    chosen, other
end

let
    kd(x) = kde(x; bandwidth=100)
    chosen, other = dwell_time(trials)
    plot()
    kdeplot!(kd(chosen), 0, 1000, line=(:black))
    kdeplot!(kd(other), 0, 1000, line=(:black, :dot))

    chosen, other = dwell_time(sim)
    kdeplot!(kd(chosen), 0, 1000, line=(RED))
    kdeplot!(kd(other), 0, 1000, line=(RED, :dot))
end


# %% ====================  ====================

function dwell_number(trials)
    chosen, fix_nums = Bool[], Int[]
    for t in trials
        nfix = counts(t.fixations, 1:3)
        for i in eachindex(nfix)
            push!(chosen, t.choice == i)
            push!(fix_nums, nfix[i])
        end
    end
    chosen, fix_nums
end
# %% ====================  ====================
fig("dwell_number") do
    plot_comparison(dwell_number, sim, :integer, :discrete)
    xticks!([0, 1], ["Unchosen", "Chosen"])
    ylabel!("Number of fixations")
end

# %% ====================  ====================
x = 5000
# plot_comparison(fixate_on_best, sim, Binning(0:x/7:x); cutoff=x)
function fix_best(trials; keep=5, min_fix=-Inf, max_fix=Inf)
    x, y = Int[], Bool[]
    for t in trials
        if min_fix <= length(t.fixations) <= max_fix
            best = argmax(t.value)
            k = min(keep, length(t.fixations))
            push!(x, (1:k)...)
            push!(y, (t.fixations .== best)[1:k]...)
        end
    end
    x, y
end

plot_comparison(fix_best, sim, :integer)
# %% ====================  ====================
function fixation_bias(trials, min_v=-Inf, max_v=Inf)
    x, y = Float64[], Bool[]
    for t in trials
        ft = total_fix_time(t)
        ft .-= mean(ft)
        for i in 1:3
            if min_v <= t.value[i] <= max_v
                push!(x, ft[i])
                push!(y, t.choice == i)
            end
        end
    end
    x, y
end

# plot_comparison(low_fixation_bias, sim)
# plot_comparison(fixation_bias, sim
V = flatten(trials.value)

fig("split_fixation_bias") do
    # a, b = quantile(V, [0.25, 0.75])
    # a, b = quantile(V, [0.1, 0.9])
    # a = b = prior[1]
    a = b = μ_emp
    hx, hy = fixation_bias(trials, -Inf, Inf)
    bins = make_bins(nothing, hx)
    plot()
    plot_human!(bins, fixation_bias(trials, -Inf, a)...)
    plot_human!(bins, fixation_bias(trials, b, Inf)...; alpha=0.3)
    plot_model!(bins, fixation_bias(sim, -Inf, a)...)
    plot_model!(bins, fixation_bias(sim, b, Inf)...; alpha=0.3)
    cross!(0, 1/3)
    xlabel!("Relative fixation time")
    ylabel!("Probability of choice")
    annotate!(75, 0.8, text("high value", :gray))
    annotate!(100, 0.1, "low value")
end

# argmin(L[4, :])
# sim1 = simulate_experiment(policies[71],  (μ_emp, σ_emp), 10)
# %% ====================  ====================


# %% ====================  ====================
function all_bad(trials, max_v=prior[1])
    x, y = Float64[], Bool[]
    for t in trials
        !all(t.value .< max_v) && continue
        ft = total_fix_time(t)
        ft .-= mean(ft)
        for i in 1:3
            push!(x, ft[i])
            push!(y, t.choice == i)
        end
    end
    x, y
end
fig("all_bad") do
    plot_comparison(all_bad, sim)
    xlabel!("Relative fixation time")
    ylabel!("Probability of choice")
end

# %% ====================  ====================

function foo()
    x, y = Float64[], Bool[]
    for i in 1:10000
        v = -ones(3) .+ rand(3)
        t = simulate(policy, v)
        ft = counts(t.samples, 1:3)
        ft = ft .- mean(ft)
        for i in 1:3
            push!(x, ft[i])
            push!(y, t.choice == i)
        end
    end
    x, y
end

mx, my = foo()
fig("model_fixation_negative") do
    bins = make_bins(nothing, mx)
    plot()
    plot_model!(bins, mx, my)
    cross!(0, 1/3)
    xlabel!("Relative fixation time")
    ylabel!("Probability of choice")
end

# %% ====================  ====================
function louie_context(trials)
    x = Float64[]
    x1 = Float64[]
    y = Bool[]
    for t in trials
        v = t.value
        s = sortperm(-v)
        t.choice == s[3] && continue  # chose one of top 2
        v[s[1]] - v[s[2]] != 2. && continue
        push!(x, v[s[3]] - v[s[2]])
        push!(y, t.choice == s[1])
        push!(x1, v[s[1]] - v[s[2]])
    end
    x, y
end

# %% ====================  ====================
@everywhere include("blinkered.jl")
@everywhere policy = $policy
@everywhere function choice_probs(v, n=1000)
    s = State(policy.m, v)
    choices = zeros(3)
    for i in 1:n
        choices[rollout(policy, state=s).choice] += 1
    end
    choices / n
end

v, p = pmap(1:96*2) do i
    v = sort!(randn(3))
    cp = choice_probs(v)
    v[1], cp[3] / cp[2]
end |> invert
scatter(v, p, ylim=(0, 10), label="")
xlabel!("Worst item value")
ylabel!("Pr(choose best) /\n Pr(choose middle)")

# %% ====================  ====================
function value_last(trials)
    x, y = Float64[], Bool[]
    for t in trials
        isempty(t.fixations) && continue
        push!(x, mean(t.value))
        push!(y, t.fixations[end] == t.choice)
    end
    x, y
end
plot_comparison(value_last, sim)

# %% ====================  ====================

function value_rt(trials)
    map(trials) do t
        mean(t.value), sum(t.fix_times)
    end |> invert
end
plot_comparison(value_rt, sim, Binning(1:1:10))

# %% ====================  ====================
 function fig3b(trials)
    map(trials) do t
        isempty(t.fixations) && return missing
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
# using StatPlots

function fixate_probs(trials; k=6)
    X = zeros(Int, k, 3)
    for t in trials
        3 < length(t.fixations) <= k || continue
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
    plot!(Xm[:, [1,3]], color=RED, label="", ls=[:solid :dash])
    xlabel!("Fixation number")
    ylabel!("Proportion of fixations")
end
# %% ====================  ====================
function fixate_probs(trials; k=4)
    X = zeros(Int, k, 3)
    for t in trials
        # length(t.fixations) != k && continue
        # length(t.fixations) < k && continue
        ranks = sortperm(-t.value)
        best, middle = t.value[ranks[1:2]]
        best - middle < 2 && continue
        for i in 1:min(k, length(t.fixations))
            r = ranks[t.fixations[i]]
            X[i, r] += 1
        end
    end
    X ./ sum(X, dims=2)
end
Xh = fixate_probs(trials)
Xm = fixate_probs(sim)

plot(Xh[:, 1], color=:black)
plot!(Xm[:, 1], line=(RED, :dash))

# %% ====================  ====================
fig("4a_alt") do
    plot(Xh[:, [1,3]], color=:black, label=["best" "worst"], ls=[:solid :dash])
    plot!(Xm[:, [1,3]], color=RED, label="", ls=[:solid :dash])
    xlabel!("Fixation number")
    ylabel!("Proportion of fixations")
end


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
    plot!(Xm[:, [1,3]], color=RED, label="", ls=[:solid :dash])
    xticks!(1:6, string.(-5:0))
    xlabel!("Fixation number")
    ylabel!("Proportion of fixations")
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

histogram(length.(sim.fixations))

sim
# %% ====================  ====================
map(sim) do t
    t.choice == argmax(t.value)
end |> mean

# %% ====================  ====================
function nth_rank(n)
    return trials -> begin
        x = Int[]
        for t in trials
            if length(t.fixations) >= n
                ranks = sortperm(sortperm(-t.value))
                push!(x, ranks[t.fixations[n]])
            end
        end
        if length(x) == 0
            return missing
        end
        cx = counts(x, 3)
        1:3, cx / sum(cx)
    end
end
plot_comparison(nth_rank(3), sim, :integer, :discrete)

# %% ====================  ====================
x = Bool[]
for t in sim
    if unique_values(t)
        ranks = sortperm(sortperm(-t.value))
        push!(x, ranks[t.fixations[1]] == 1)
    end
end
# %% ====================  ====================
success, n = sum(x), length(x)
success/n
using RCall
R"binom.test($success, $n, 1/3)"

# %% ====================  ====================

@everywhere function nfix_by_time(trials)
    x, y = Float64[], Int[]
    for t in trials
        for (i, ft) in enumerate(cumsum(t.fix_times))
            push!(x, ft)
            push!(y, i)
        end
    end
    x, y
end

fig("nfix_by_time") do
    plot_comparison(nfix_by_time, sim, Binning(0:100:3000))
    xlabel!("Time (ms)")
    ylabel!("Number of fixations")
end

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
