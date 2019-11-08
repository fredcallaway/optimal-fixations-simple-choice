include("plots_base.jl")
include("pseudo_likelihood.jl")
include("params.jl")
include("box.jl")
using Glob
using StatsBase
plot([1,2])

# %% ==================== Joint fitting results ====================
run_name = "joint_fit"
both_trials = load_dataset.(["two", "three"])
# res = get_result("results/both_items_fixed/2019-10-26T16-32-24-O5k/")
# res = get_result("results/both_items_fixed_parallel_post/2019-10-28T12-31-17-5Zc/")
res = get_result("results/test_post/2019-10-28T16-01-38-Hly/")
prm = load(res, :mle)


res = get_result("results/fit_pseudo/2019-11-06T00-07-11-Gsg")
load(res, :mle)

res = get_result("results/fit_pseudo/2019-11-06T00-07-11-2Ga")
load(res, :mle)

function train_test_split(trials, fold)
    train_idx = Dict(
        "odd" => 1:2:length(trials),
        "even" => 2:2:length(trials),
        "all" => 1:length(trials),
    )[fold]
    test_idx = setdiff(eachindex(trials), train_idx)
    (train=trials[train_idx], test=trials[test_idx])
end

fold = load(res, :args)["fold"]
both_trials = map(["two", "three"]) do n
    trials = load_dataset(n)
    # train_test_split(trials, fold).test
end

empirical_prior(trials) = juxt(mean, std)(flatten(trials.value))
emp_priors = map(both_trials) do trials
    empirical_prior(trials)
end


# %% ====================  ====================
reopt = load(res, :reopt);
both_sims = map(1:2) do i
    trials = both_trials[i]
    μ_emp, σ_emp = empirical_prior(trials)
    policies = reopt[i].policies
    asyncmap(policies) do pol
        simulate_experiment(pol, trials; μ=prm.β_μ * μ_emp, σ=prm.β_σ * σ_emp,
            sample_time=prm.sample_time, n_repeat=10)
    end
end;
length.(both_sims)

# %% ====================  ====================
function make_lines!(xline, yline, trials)
    if xline != nothing
        vline!([xline], line=(:grey, 0.7))
    end
    if yline != nothing
        if yline == :chance
            # yline = 1 / n_item(trials[1])
            yline = 1 / length(trials[1].value)
        end
        hline!([yline], line=(:grey, 0.7))
    end
end


function plot_one(feature, xlab, ylab, trials, sims, plot_kws=();
        binning=nothing, type=:line, xline=nothing, yline=nothing,
        save=false, name=string(feature), kws...)
    hx, hy = feature(trials; kws...)
    bins = make_bins(binning, hx)
    f = plot(xlabel=xlab, ylabel=ylab; plot_kws...)

    plot_human!(bins, hx, hy, type)
    for sim in sims
        mx, my = feature(sim; kws...)
        plot_model!(bins, mx, my, type, alpha=0.5)
    end
    make_lines!(xline, yline, trials)
    if save
        savefig(f, "figs/$run_name/$name.pdf")
    end
    f
end

function plot_one(name::String, xlab, ylab, trials, sims, plot_kws;
        xline=nothing, yline=nothing,
        plot_human::Function, plot_model::Function)

    f = plot(xlabel=xlab, ylabel=ylab; plot_kws...)

    plot_human(trials)
    for sim in sims
        plot_model(sim)
    end
    make_lines!(xline, yline, trials)
    f
end

# left_rv = n_item == 2 ? "Left rating - right rating" : "Left rating - mean other rating"
# best_rv = n_item == 2 ? "Best rating - worst rating" : "Best rating - mean other rating"
DISABLE_ALIGN = true
function plot_both(feature, xlab, ylab, plot_kws=(); align=:default, name=string(feature), kws...)
    xlab1, xlab2 =
        (xlab == :left_rv) ? ("Left rating - right rating", "Left rating - mean other rating") :
        (xlab == :best_rv) ? ("Best rating - worst rating", "Best rating - mean other rating") :
        (xlab, xlab)

    # return plot_one(feature, xlab, ylab, trials, sims, plot_kws; kws...)


    f1 = plot_one(feature, xlab1, ylab, both_trials[1], both_sims[1], plot_kws; kws...)
    f2 = plot_one(feature, xlab2, ylab, both_trials[2], both_sims[2], plot_kws; kws...)

    ylabel!(f2, "  ")
    x1 = xlims(f1); x2 = xlims(f2)
    y1 = ylims(f1); y2 = ylims(f2)
    for (i, f) in enumerate([f1, f2])
        if align != :y_only
            xlims!(f, min(x1[1], x2[1]), max(x1[2], x2[2]))
        end
        if align == :default || DISABLE_ALIGN
            ylims!(f, min(y1[1], y2[1]), max(y1[2], y2[2]))
        elseif align == :chance
            rng = max(maximum(abs.(y1 .- 1/2)), maximum(abs.(y2 .- 1/3)))
            chance = [1/2, 1/3][i]
            ylims!(f, chance - rng, chance + rng)
        end
    end
    ff = plot(f1, f2, size=(900,400))

    if haskey(Dict(kws), :fix_select)
        name *= "_$(kws[:fix_select])"
    end
    savefig(ff, "figs/$run_name/$name.pdf")
    nothing
end

include("features.jl")

# %% ==================== Basic psychometrics ====================
# run_name = "indiv2"
run_name = "nov8_both_400"
mkpath("figs/$run_name")

plot_both(value_choice, :left_rv, "P(left chosen)";
    xline=0, yline=:chance, binning=Binning(-4.5:1:4.5))

plot_both(difference_time, :best_rv, "Total fixation time [ms]")

plot_both("rt_kde", "Total fixation time [ms]", "Density", (yticks=[],),
    plot_human=(trials)->kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=6000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=6000, line=(RED, 2, 0.5))
)

# %% ==================== Number of fixations ====================
plot_both(nfix_hist, "Number of fixations", "Proportion of trials",
    (xticks=[1,5,10], ),
    binning=:integer, type=:discrete)

plot_both(difference_nfix, :best_rv, "Number of fixations",
    binning=:integer)

# %% ==================== Fixation locations ====================

plot_both(fixate_on_best, "Time since trial onset [ms]", "P(fixate best)",
    (xticks=(0.5:2:8.5, string.(0:500:2000)), xlims=(0,8.5)),
    binning=:integer, yline=:chance, align=:chance,
    cutoff=2000, n_bin=8)

plot_both(value_bias, :left_rv, "Proportion fixation time";
    xline=0, yline=:chance)

plot_both("refixate_uncertain", "Fixation advantage\n of refixated item [ms]", "Density",
    (yticks=[],),
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials), 100., xmin=-1000, xmax=1000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(refixate_uncertain(sim), 100., xmin=-1000, xmax=1000, line=(RED, 2, 0.5)),
    xline=0
)

# %% ==================== Fixation durations ====================

plot_both(binned_fixation_times, "Fixation type", "Fixation duration",
    (xticks=(1:4, ["first", "second", "middle", "last"]),),
    binning=:integer, type=:discrete)

plot_both(full_fixation_times, "Fixation number", "Fixation duration",
    binning=:integer)

plot_both(chosen_fix_time, "", "Average fixation duration",
    (xticks=(0:1, ["Unchosen", "Chosen"]),),
    binning=:integer, type=:discrete; fix_select=nonfinal)

plot_both(value_duration, "Item value", "Fixation duration",
    binning=:integer; fix_select=firstfix)


# %% ==================== Last fixations ====================
plot_both(value_duration, "Item value",  "Fixation duration [ms]",
    binning=:integer, fix_select=final)

plot_both(last_fixation_duration, "Chosen item time advantage\nbefore last fixation [ms]",
    # (xticks=[],),
    "Last fixation duration [ms]")

# %% ==================== Mechanism tests for 3 items ====================

plot_one(fix4_value, "Rating of first minus second fixated item",
    "P(4th fixation is refixation\nto first fixated item)",
    both_trials[2], both_sims[2], (xticks=-6:2:6,), xline=0, save=true)

plot_one(fix4_uncertain,
    "First minus second fixation duration [ms]",
    "P(4th fixation is refixation\nto first fixated item)",
    both_trials[2], both_sims[2], xline=0, save=true)

plot_one(fix3_value,
    "Rating of first fixated item",
    "P(3rd fixation is refixation\nto first fixated item)",
    both_trials[2], both_sims[2], save=true)

plot_one(fix3_uncertain,
    "Duration of first fixation",
    "P(3rd fixation is refixation\nto first fixated item)",
    both_trials[2], both_sims[2], save=true)


# %% ==================== Choice biases ====================
plot_both(last_fix_bias, "Last fixated item relative rating", "P(last fixated item chosen)",
    binning=:integer; xline=0, yline=:chance)

plot_both(fixation_bias, "Final time advantage left [ms]", "P(left chosen)",
    ; xline=0, yline=:chance,
    # trial_select=(t)->t.value[1] == 3
    )


plot_both(fixation_bias_corrected, "Final time advantage left [ms]", "corrected P(left chosen)",
    ; xline=0, yline=0)

plot_both(first_fixation_duration, "First fixation duration [ms]", "P(first fixated chosen)",
    )

# %% ====================  ====================

ranks = sortperm(sortperm(t.value; rev=true))
ranks[t.fixations[1]]

map(trials) do t




# %% ====================  ====================

function fixate_on_(trials, which; sample_time=10, cutoff=2000, n_bin=8, nonfinal=false)
    n_sample = Int(cutoff / sample_time)
    spb = Int(n_sample/n_bin)
    x = Int[]
    y = Float64[]
    for t in trials
        unique_values(t) || continue
        fix = discretize_fixations(t; sample_time=sample_time)
        # fix = fix[1:n_sample]
        ns = min(n_sample, length(fix))
        fix_best = fix[1:ns] .== which(t.value)
        props = mean.(Iterators.partition(fix_best, spb))
        if nonfinal
            props = props[1:end-1]
        end

        push!(x, eachindex(props)...)
        push!(y, props...)
    end
    x, y
end

plot_both(fixate_on_best, "Time since trial onset [ms]", "P(fixate best)",
    # (xticks=(0.5:2:10.5, string.(0:500:2000)), xlims=(0,8.5)),
    binning=:integer, yline=:chance, align=:chance,
    cutoff=3000, n_bin=10)
