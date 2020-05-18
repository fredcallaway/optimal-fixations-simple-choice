if basename(PROGRAM_FILE) == basename(@__FILE__)
    INTERACTIVE = false
    run_name = ARGS[1]
else
    INTERACTIVE = true
    run_name = "final"
end
include("plots_base.jl")

RUN_NAMES = ["main14"]
out_path = "figs/main14"

RUN_NAMES = ["main14", "rando17"]
out_path = "figs/compare_rando"

RUN_NAMES = ["main14", "lesion19"]
out_path = "figs/compare_lesion"
#
# RUN_NAMES = ["main14", "lesion17", "rando17"]
# out_path = "figs/compare_all"
# mkpath(out_path)

rm(out_path, recursive=true, force=true)
mkpath(out_path)

RUN_NAMES

# %% ==================== Basic psychometrics ====================

plot_both(value_choice, :left_rv, "P(left chosen)";
    xline=0, yline=:chance, binning=Binning(-4.5:1:4.5))

function rt_kde(trials; kws...)
    kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=6000; kws...)
end
plot_both(rt_kde, "Total fixation time [ms]", "Density"; yticks=false, manual=true)

plot_both(difference_time, :best_rv, "Total fixation time [ms]",
    binning=:integer)


# %% ==================== Baisc fixation properties ====================

plot_both(nfix_hist, "Number of fixations", "Proportion of trials",
    (xticks=[1,5,10], ),
    binning=:integer, type=:discrete)

plot_both(difference_nfix, :best_rv, "Number of fixations",
    binning=:integer)  # FIXME binning is weird

plot_both(binned_fixation_times, "Fixation number", "Fixation duration [ms]",
    (xticks=(1:7, ["1", "2", "3", "4", "5", ">5", "final"]),),
    binning=:integer)

savefig(ans, "$out_path/$binned_fixation_times.pdf")



# %% ==================== Uncertainty-directed attention ====================

function plot_uncertain(trials; kws...)
    kdeplot!(refixate_uncertain(trials), 100., xmin=-1000, xmax=1000,; kws...)
end

plot_both(plot_uncertain, "Fixation advantage\n of fixated item [ms]", "Density",
    yticks = false, name="refixate_uncertain", manual=true,
    xline=0
)

function plot_uncertain_alt(trials; kws...)
    kdeplot!(refixate_uncertain(trials, ignore_current=true), 100., xmin=-1000, xmax=1000,; kws...)
end

plot_three(plot_uncertain_alt, "Alternative fixation advantage\nof fixated item [ms]", "Density",
    xline=0, yticks = false, name="refixate_uncertain_alt", manual=true,
)

plot_three(fixate_by_uncertain, "Alternative fixation advantage\nof more-fixated item [ms]", "P(fixate more-fixated item)",
    yline=1/2, binning=Binning(-50:100:850))

# %% ==================== Value-directed attention ====================

plot_both(value_bias, :left_rv, "Proportion fixate left";
    binning=:integer, xline=0, yline=:chance)

plot_both(value_duration, "First fixated item rating",  "First fixation duration [ms]",
    binning=:integer, fix_select=firstfix)

plot_both(fixate_on_worst, "Cumulative fixation time [ms]", "P(fixate worst)",
    (xticks=(0.5:5:20.5, string.(0:500:2000)), xlims=(0,20.5)),
    binning=:integer, yline=:chance, align=:chance,
    cutoff=2000, n_bin=20)

plot_three(fix4_value, "Rating of first minus\nsecond fixated item",
    "P(4th fixation is refixation\nto first fixated item)",
    binning=:integer,
    (xticks=-6:2:6,), xline=0)

plot_three(fix3_value,
    binning=:integer,
    "Rating of first fixated item",
    "P(third fixation is refixation\nto first fixated item)")

# %% ==================== Choice biases ====================
plot_both(last_fix_bias, :last_rv, "P(last fixated item chosen)",
    binning=:integer, xline=0, yline=:chance)

plot_both(fixation_bias, "Final time advantage left [ms]", "P(left chosen)",
    ; xline=0, yline=:chance,
    # trial_select=(t)->t.value[1] == 3
    )

plot_both(first_fixation_duration, "First fixation duration [ms]", "P(first fixated chosen)",
    yline= :chance )

plot_both(fixation_bias_corrected, "Final time advantage left [ms]", "corrected P(left chosen)",
    xline=0, yline=0)

plot_both(first_fixation_duration_corrected, "First fixation duration [ms]", "corrected P(first fixated chosen)",
    yline=0 )
