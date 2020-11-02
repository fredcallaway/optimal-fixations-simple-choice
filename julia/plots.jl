INTERACTIVE = basename(PROGRAM_FILE) == ""

include("plots_base.jl")

PARAMS = [
    # (run_name="revision", dataset="joint", prior="zero", color=(colorant"#699efa", colorant"#003085"), alpha=0.1, n_sim=30),
    (run_name="revision", dataset="joint", prior="fit", color=(colorant"#ff6167", colorant"#b00007"), alpha=1, n_sim=30),
    (run_name="addm", color=(colorant"#4DCE3F", colorant"#3B9B31"), alpha=1, n_sim=1),
]
out_path = "figs/revision/addm2"

rm(out_path, recursive=true, force=true)
mkpath(out_path)


# %% ==================== Basic psychometrics ====================

plot_both(value_choice, :left_rv, "P(left chosen)";
    xline=0, yline=:chance, binning=Binning(-4.5:1:4.5))

function rt_kde(trials; kws...)
    kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=6000; kws...)
end
plot_both(rt_kde, "Total fixation time [ms]", "Density"; yticks=false, manual=true)

plot_both(difference_time, :best_rv, "Total fixation time [ms]",
    binning=:integer)


# %% ==================== Basic fixation properties ====================

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
    kdeplot!(refixate_uncertain(trials), 100., xmin=-1000, xmax=1000; kws...)
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


# %% ==================== Individuals ====================

function plot_individuals(feature, xlab, ylab, plot_kws=(); n_col=5, kws...) 
    foreach([2, 3]) do n_item
        xlab = fmt_xlab(xlab, n_item)
        subjects = both_trials[n_item - 1].subject |> unique
        plots = map(subjects) do subj
            plot_one(feature, n_item, xlab, ylab, plot_kws; subject=subj, kws...)
        end
        
        n_row = cld(length(plots), n_col)
        empty_plot = plot(axis=false, grid=false)
        pp = reshape(deepcopy([plots; fill(empty_plot, n_row * n_col - length(plots))]), n_row, n_col)

        for p in pp[:, 2:end]
            plot!(p, yformatter=_->"", ylabel="")
        end
        for p in pp[1:end-1, :]
            plot!(p, xformatter=_->"", xlabel="")
        end
        for i in 1:n_row
            i != cld(n_row,2) && plot!(pp[i, 1], ylabel="")
        end
        for i in 1:n_col
            i != cld(n_col,2) && plot!(pp[end, i], xlabel="")
        end
        plot(permutedims(pp)..., layout=(n_row,n_col), size=(n_col*100, n_row * 100))
        name = string(feature)
        savefig("figs/indiv/$name-$n_item.pdf")
    end
end

# %% --------
plot_individuals(value_bias, :left_rv, "Proportion fixate left",
    (ylim=(-0.05, 1.05),);
    xline=0, yline=:chance, binning=Binning(-5:2:5), plot_model=false,)
# %% --------
plot_individuals(value_choice, :left_rv, "P(left chosen)",
    (ylim=(-0.05, 1.05),);
    xline=0, yline=:chance, binning=Binning(-5:2:5), plot_model=false)
# %% --------
plot_individuals(binned_fixation_times, "Fixation number", "Fixation duration [ms]",
    (ylim=(0, 1300), xticks=(1:7, ["1", "2", "3", "4", "5", ">5", "F"]),),
    binning=:integer, plot_model=false)

