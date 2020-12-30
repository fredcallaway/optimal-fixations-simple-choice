INTERACTIVE = basename(PROGRAM_FILE) == ""
include("plots_base.jl")
# RELOAD = true
# %% --------

blue = colorant"#417CFF"  # unbiased
pink = colorant"#FA60BE"  # zero
purple = colorant"#8D58D7"
light_purple = colorant"#DDB3FF"
green = colorant"#65B55F"
black = colorant"black"

PARAMS = [
    (id=:zero, total_only=true, run_name="revision", dataset="joint", prior="zero", color=(black, blue), alpha=0.6, n_sim=30, plot_kws=(;linestyle=:solid)),
    (id=:unbiased, total_only=true, run_name="revision", dataset="joint", prior="unbiased", color=(black, pink), alpha=0.6, n_sim=30, plot_kws=(;linestyle=:solid)),
    (id=:fit, total_only=false, run_name="revision", dataset="joint", prior="fit", color=(light_purple, purple), alpha=0.6, n_sim=30, plot_kws=()),
    (id=:addm, total_only=true, run_name="addm", color=(black, green), alpha=0.6, n_sim=1, plot_kws=(;linestyle=:dash)),
]

out_path = "figs/revision/all"
# out_path = "figs/revision/tmp"
# rm(out_path, recursive=true, force=true)
mkpath(out_path)

# %% ==================== Basic psychometrics ====================
plot_both(value_choice, :left_rv, "P(left chosen)",
    (xticks=[-5,0,5],),
    xline=0, yline=:chance, 
    binning=:integer
    )

function plot_quantiles!(x, id, scale=1; kws...)
    heights = Dict(zip([:addm, :human, :fit,], .00002 .* (0:10)))
    qs = quantile(x, 0:0.25:1)[2:end-1]
    scatter!(qs, marker=([7, 7], [:circle, :square]), scale .* heights[id] .* ones(length(qs)); markerstrokealpha=0, kws...)
end

function rt_kde(trials; id, total=true, kws...)
    rt = sum.(trials.fix_times)
    kdeplot!(rt, 300., xmin=0, xmax=8000; kws...)
    if total && id ∈ [:addm, :fit, :human]
        plot_quantiles!(rt, id; kws...)
    end
end

plot_both(rt_kde, "Total fixation time [ms]", "Density",
    (xticks=[0,2000,4000,6000],),
    ; yticks=false, manual=true)

plot_both(difference_time, :best_rv, "Total fixation time [ms]",
    binning=:integer)

plot_both(meanvalue_time, "Mean item rating", "Total fixation time [ms]")


# %% ==================== Basic fixation properties ====================

plot_both(nfix_hist, "Number of fixations", "Proportion of trials",
    (xticks=[1,5,10], ),
    binning=:integer, type=:discrete)

plot_both(difference_nfix, :best_rv, "Number of fixations",
    binning=:integer)  # FIXME binning is weird

plot_both(binned_fixation_times, "Fixation number", "Fixation duration [ms]",
    (xticks=(1:7, ["1", "2", "3", "4", "5", ">5", "final"]),),
    binning=:integer)

# %% ==================== Uncertainty-directed attention ====================

function plot_uncertain(trials; id, total=true, kws...)
    xs = refixate_uncertain(trials)
    kdeplot!(xs, 100., xmin=-1000, xmax=1000; kws...)
    if total && id ∈ [:addm, :fit, :human]
        plot_quantiles!(xs, id, 5; kws...)
    end
end

plot_both(plot_uncertain, "Fixation advantage\n of fixated item [ms]", "Density",
    yticks = false, name="refixate_uncertain", manual=true,
    xline=0
)

function plot_uncertain_alt(trials; id, total=true, kws...)
    xs = refixate_uncertain(trials, ignore_current=true)
    kdeplot!(xs, 100., xmin=-1000, xmax=1000; kws...)
    if total && id ∈ [:addm, :fit, :human]
        plot_quantiles!(xs, id, 5; kws...)
    end
end

plot_three(plot_uncertain_alt, "Alternative fixation advantage\nof fixated item [ms]", "Density",
    xline=0, yticks = false, name="refixate_uncertain_alt", manual=true,
)

plot_three(fixate_by_uncertain, "Alternative fixation advantage\nof more-fixated item [ms]", "P(fixate more-fixated item)",
    yline=1/2, binning=7)



# %% ==================== Value-directed attention ====================

plot_both(value_bias, :left_rv, "Proportion fixate left",
    (xticks=[-5, 0, 5],),
    binning=:integer, xline=0, yline=:chance)

plot_both(value_duration, "First fixated item rating",  "First fixation duration [ms]",
    (xticks=[0,5,10],),
    binning=:integer, fix_select=firstfix)

plot_both(fixate_on_worst, "Cumulative fixation time [ms]", "P(fixate worst)",
    (xticks=(0.5:5:20.5, string.(0:500:2000)), xlims=(0,20.5)),
    binning=:integer, yline=:chance, align=:chance,
    cutoff=2000, n_bin=20)

# plot_both(fixate_on_worst, "Cumulative fixation time [ms]", "P(fixate worst)",
#     (xticks=(0.5:5:20.5, string.(0:500:2000)), xlims=(0,20.5)),
#     binning=:integer, yline=:chance, align=:chance,
#     cutoff=2000, n_bin=20)

plot_three(fix4_value, "Rating of first minus\nsecond fixated item",
    "P(4th fixation is refixation\nto first fixated item)",
    binning=:integer,
    (xticks=-6:2:6,), xline=0)

plot_three(fix3_value,
    "Rating of first fixated item",
    "P(third fixation is refixation\nto first fixated item)",
    binning=:integer,
    )

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

