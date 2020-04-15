using Distributed

if basename(PROGRAM_FILE) == basename(@__FILE__)
    INTERACTIVE = false
    run_name = ARGS[1]
else
    INTERACTIVE = true
    run_name = "final"
end

println(run_name)

@everywhere begin
    run_name = $run_name
    include("plots_base.jl")
end
length(both_sims[1])
length(both_sims[1][1])
plot([1,2])

# %% ==================== Basic psychometrics ====================

plot_both(value_choice, :left_rv, "P(left chosen)";
    xline=0, yline=:chance, binning=Binning(-4.5:1:4.5))

plot_both("rt_kde", "Total fixation time [ms]", "Density"; yticks=false,
    plot_human=(trials)->kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=6000, line=(:black, 2)),
    plot_model=(sim; color=RED)->kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=6000, line=(color, 2, ALPHA)) )

plot_both(difference_time, :best_rv, "Total fixation time [ms]",
    binning=:integer)

# %% ==================== Baisc fixation properties ====================

plot_both(nfix_hist, "Number of fixations", "Proportion of trials",
    (xticks=[1,5,10], ),
    binning=:integer, type=:discrete)

plot_both(difference_nfix, :best_rv, "Number of fixations",
    binning=:integer)  # FIXME binning is weird

plot_both(binned_fixation_times, "Fixation number", "Fixation duration [ms]",
    (xticks=(1:7, ["1", "2", "3", "4", "5", ">5", "last"]),),
    binning=:integer)

# %% ==================== Uncertainty-directed attention ====================

plot_both("refixate_uncertain", "Fixation advantage\n of fixated item [ms]", "Density",
    yticks = false,
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials), 100., xmin=-1000, xmax=1000, line=(:black, 2)),
    plot_model=(sim; color=RED)->kdeplot!(refixate_uncertain(sim), 100., xmin=-1000, xmax=1000, line=(color, 2, ALPHA)),
    xline=0
)

plot_three("refixate_uncertain_alt", "Alternative fixation advantage\nof fixated item [ms]", "Density",
    (yticks = [],),
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials, ignore_current=true), 100., xmin=-1000, xmax=1000, line=(:black, 2)),
    plot_model=(sim; color=RED)->kdeplot!(refixate_uncertain(sim, ignore_current=true), 100., xmin=-1000, xmax=1000, line=(color, 2, ALPHA)),
    xline=0
)

plot_three(fixate_by_uncertain, "Alternative fixation advantage\nof more-fixated item", "P(fixate more-fixated item)", yline=1/2,
    binning=Binning(-50:100:850))

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
    "P(3rd fixation is refixation\nto first fixated item)")


# %% ==================== Choice biases ====================
plot_both(last_fix_bias, :last_rv, "P(last fixated item chosen)",
    binning=:integer, xline=0, yline=:chance)

plot_both(fixation_bias, "Final time advantage left [ms]", "P(left chosen)",
    ; xline=0, yline=:chance,
    # trial_select=(t)->t.value[1] == 3
    )
#
# plot_both(fixation_bias_corrected, "Final time advantage left [ms]", "corrected P(left chosen)",
#     xline=0, yline=0)

plot_both(first_fixation_duration, "First fixation duration [ms]", "P(first fixated chosen)",
    yline= :chance )

plot_both(first_fixation_duration_corrected, "First fixation duration [ms]", "corrected P(first fixated chosen)",
    yline=0 )
