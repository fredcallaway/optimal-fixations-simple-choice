include("plots_base.jl")
include("human.jl")
using Glob
using StatsBase
plot([1,2])

# %% ====================  ====================

both_trials = map(["two", "three"]) do num
    load_dataset(num)[1:2:end]  # out of sample prediction
end

run_name = "sobol4"
fit_mode = "separate"
fit_prior = "true"
out_path = "figs/$run_name/$fit_mode-$fit_prior"
mkpath("$out_path")

both_sims = map(1:30) do i
    map(deserialize("results/$run_name/simulations/$fit_mode-$fit_prior/$i")) do sims
        reduce(vcat, sims)
    end
end |> invert;


NO_RIBBON = false
SKIP_BOOT = true
# %% ==================== Basic psychometrics ====================
# run_name = "no_inner"
# both_sims = [ [ms[i] for ms in multi_sims] for i in 1:2 ];
plot_both(value_choice, :left_rv, "P(left chosen)";
    xline=0, yline=:chance, binning=Binning(-4.5:1:4.5))

plot_both(difference_time, :best_rv, "Total fixation time [ms]",
    binning=:integer)

plot_both("rt_kde", "Total fixation time [ms]", "Density"; yticks=false,
    plot_human=(trials)->kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=6000, line=(:black, 2)),
    plot_model=(sim; color=RED)->kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=6000, line=(color, 2, 0.5))
)

# %% ==================== Number of fixations ====================

plot_both(nfix_hist, "Number of fixations", "Proportion of trials",
    (xticks=[1,5,10], ),
    binning=:integer, type=:discrete)

plot_both(difference_nfix, :best_rv, "Number of fixations",
    binning=:integer)  # FIXME binning is weird

# %% ==================== Fixation locations ====================

plot_both(fixate_on_best, "Time since trial onset [ms]", "P(fixate best)",
    (xticks=(0.5:2:8.5, string.(0:500:2000)), xlims=(0,8.5)),
    binning=:integer, yline=:chance, align=:chance,
    cutoff=2000, n_bin=8)

plot_both(fixate_on_worst, "Time since trial onset [ms]", "P(fixate worst)",
    (xticks=(0.5:2:8.5, string.(0:500:2000)), xlims=(0,8.5)),
    binning=:integer, yline=:chance, align=:chance,
    cutoff=2000, n_bin=8)

plot_both(value_bias, :left_rv, "Proportion fixate left";
    xline=0, yline=:chance)

plot_both("refixate_uncertain", "Fixation advantage\n of refixated item [ms]", "Density",
    yticks = false,
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials), 100., xmin=-1000, xmax=1000, line=(:black, 2)),
    plot_model=(sim; color=RED)->kdeplot!(refixate_uncertain(sim), 100., xmin=-1000, xmax=1000, line=(color, 2, 0.5)),
    xline=0
)

# %% ==================== Fixation durations ====================

plot_both(binned_fixation_times, "Fixation type", "Fixation duration [ms]",
    (xticks=(1:4, ["first", "second", "middle", "last"]),),
    binning=:integer, type=:discrete)

# plot_both(full_fixation_times, "Fixation number", "Fixation duration [ms]",
#     binning=Binning(0.5:1:9.5))

plot_both(chosen_fix_time, "", "Average fixation duration [ms]",
    (xticks=(0:1, ["Unchosen", "Chosen"]),),
    binning=:integer, type=:discrete; fix_select=nonfinal)

plot_both(value_duration, "Item value",  "Fixation duration [ms]",
    binning=:integer, fix_select=firstfix)

# %% ==================== Last fixations ====================

# plot_both(value_duration, "Item value",  "Fixation duration [ms]",
#     binning=:integer, fix_select=final)

plot_both(last_fixation_duration, "Chosen item time advantage\nbefore last fixation [ms]",
    # (xticks=[],),
    "Last fixation duration [ms]")

# %% ==================== Mechanism tests for 3 items ====================

plot_one(fix4_value, "Rating of first minus second fixated item",
    binning=:integer,
    "P(4th fixation is refixation\nto first fixated item)",
    both_trials[2], both_sims[2], (xticks=-6:2:6,), xline=0, save=true)

plot_one(fix3_value,
    binning=:integer,
    "Rating of first fixated item",
    "P(3rd fixation is refixation\nto first fixated item)",
    both_trials[2], both_sims[2], save=true)

# %% ==================== Choice biases ====================

plot_both(last_fix_bias, :last_rv, "P(last fixated item chosen)",
    binning=:integer, xline=0, yline=:chance)

plot_both(fixation_bias, "Final time advantage left [ms]", "P(left chosen)",
    ; xline=0, yline=:chance,
    # trial_select=(t)->t.value[1] == 3
    )


plot_both(fixation_bias_corrected, "Final time advantage left [ms]", "corrected P(left chosen)",
    ; xline=0, yline=0)


plot_both(first_fixation_duration, "First fixation duration [ms]", "P(first fixated chosen)",
    )
