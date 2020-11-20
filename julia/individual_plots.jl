INTERACTIVE = basename(PROGRAM_FILE) == ""

include("plots_base.jl")
purple = colorant"#6B00D6"
light_purple = colorant"#DCB8FF"

PARAMS = [
    (id=:fit, total_only=false, run_name="revision", dataset="joint", prior="fit", color=(light_purple, purple), alpha=0.3, n_sim=30),
]
out_path = "figs/revision/individual-bayes"

# rm(out_path, recursive=true, force=true)
# mkpath(out_path)

# %% --------

function get_lims(feature, n_item, binning, kws)
    hx, hy = feature(trials; kws...)
    bins = get_bins(feature, n_item, binning, kws)
    by = mean.(bin_by(bins, hx, hy))

    (xlim=expand(bins.limits[1], bins.limits[end]),
     ylim=expand(minimum(by), maximum(by)))
end
FAST = false
function plot_both_ind(feature, xlab, ylab, xlim=nothing, ylim=nothing, plot_kws=(); name = string(feature), three_only=false, n_col=5, kws...)
    # kws = (kws..., plot_model=false)

    for n_item in (three_only || FAST ? [3] : [2, 3])

        # feature = difference_time; n_item = 2; binning = :integer; feature_kws = (); plot_kws = (); kws = (); xlab=""; ylab=""

        xlab = fmt_xlab(xlab, n_item)
        subjects = both_trials[n_item - 1].subject |> unique

        S = both_trials[n_item - 1].subject
        sc = countmap(S)
        subjects = filter(s->sc[s] == 50, unique(S))
        subjects = subjects[1:min(length(subjects), 30)] |> sort

        if FAST
            subjects = subjects[1:1]
        end
        
        plots = map(subjects) do subj
            plot_one(feature, n_item, xlab, ylab, 
                # (;get_lims(feature, n_item, binning, feature_kws)..., plot_kws...),
                (;xticks=collect(xlim), yticks=collect(ylim), xlims=expand(xlim...), ylims=expand(ylim...), 
                    right_margin=9mm, bottom_margin=1mm, title=subj, plot_kws...)
                # (plot_kws..., title=subj); 
                ; subject=subj, kws...)
        end

        # xlo, xhi = invert(xlims.(plots))
        # xlo = minimum(xlo); xhi = maximum(xhi)
        # ylo, yhi = invert(ylims.(plots))
        # ylo = minimum(ylo); yhi = maximum(yhi)
        # for p in plots
        #     xlims!(p, xlo, xhi)
        #     ylims!(p, ylo, yhi)
        # end
        
        n_row = cld(length(plots), n_col)

        empty_plot = plot(axis=false, grid=false)
        while length(plots) < n_row * n_col
            push!(plots, deepcopy(empty_plot))
        end

        pp = permutedims(reshape(plots, n_col, n_row))

        for p in pp[:, 2:end]
            plot!(p, yticks=[], ylabel="")
        end
        for p in pp[1:end-1, :]
            plot!(p, xticks=[], xlabel="")
        end
        for i in 1:n_row
            i != cld(n_row,2) && plot!(pp[i, 1], ylabel="")
        end
        for i in 1:n_col
            i != cld(n_col,2) && plot!(pp[end, i], xlabel="")
        end

        ff = plot(collect(permutedims(pp))..., layout=(n_row,n_col), size=(n_col*200, n_row*200))

        savefig(ff, "$out_path/$name-$n_item.pdf")
        # run(`open $out_path/$name-$n_item.pdf`)
    end
end

plot_three_ind(args...; kws...) = plot_both_ind(args...; kws..., three_only=true)


# %% ==================== Basic psychometrics ====================
plot_both_ind(value_choice, :left_rv, "P(left chosen)",
    (-6, 6), (0, 1)
    ; xline=0, yline=:chance, binning=:integer)

function rt_kde(trials; kws...)
    kdeplot!(sum.(trials.fix_times), 500., xmin=0, xmax=6000; kws...)
end

plot_both_ind(rt_kde, "Total fixation time [ms]", "Density",
    (0, 6000), (0, .001),
    (yticks = false,),
    manual=true)

plot_both_ind(difference_time, :best_rv, "Total fixation time [ms]",
    (0, 6), (0, 10000),
    binning=:integer)

plot_both_ind(meanvalue_time, "Mean item rating", "Total fixation time [ms]",
    (0, 8), (0, 10000),
    binning=:integer)


# %% ==================== Basic fixation properties ====================

plot_both_ind(nfix_hist, "Number of fixations", "Proportion of trials",
    (1, 10), (0, 1),
    (xlims=(0, 10.5),),
    ; binning=:integer, type=:discrete)

plot_both_ind(difference_nfix, :best_rv, "Number of fixations",
    (0, 6), (0, 8),
    binning=:integer)  # FIXME binning is weird

plot_both_ind(binned_fixation_times, "Fixation number", "Fixation duration [ms]",
    (1, 5), (0, 2000),
    (xticks=(1:5, ["1", "2", "3", "4", "5"]),),
    binning=:integer, x_max=5)


# %% ==================== Uncertainty-directed attention ====================

function plot_uncertain(trials; kws...)
    kdeplot!(refixate_uncertain(trials), 300., xmin=-2000, xmax=2000; kws...)
end

plot_both_ind(plot_uncertain, "Fixation advantage\n of fixated item [ms]", "Density",
    (-2000, 2000), (0, .002),
    (yticks = false,), 
    name="refixate_uncertain", manual=true,
    xline=0
)

function plot_uncertain_alt(trials; kws...)
    kdeplot!(refixate_uncertain(trials, ignore_current=true), 100., xmin=-1000, xmax=1000,; kws...)
end

plot_three_ind(plot_uncertain_alt, "Alternative fixation advantage\nof fixated item [ms]", "Density",
    (-1000, 1000), (0, .002),
    (yticks = false,),
    xline=0, yticks = false, name="refixate_uncertain_alt", manual=true,
)

plot_three_ind(fixate_by_uncertain, "Alternative fixation advantage\nof more-fixated item [ms]", "P(fixate more-fixated item)",
    (0, 1500), (0, 1),
    yline=1/2, binning=7)

# %% ==================== Value-directed attention ====================

plot_both_ind(value_bias, :left_rv, "Proportion fixate left",
    (-6, 6), (0, 1),
    ; binning=:integer, xline=0, yline=:chance)

plot_both_ind(value_duration, "First fixated item rating",  "First fixation duration [ms]",
    (0, 10), (0, 2000),
    ; fix_select=firstfix, binning=:integer)

plot_both_ind(fixate_on_worst, "Cumulative fixation time [ms]", "P(fixate worst)",
    (0, 20), (0, 1),
    (xticks=([0, 20], ["0", "2000"]), xlims=(0,20.5)),
    ; cutoff=2000, n_bin=20, binning=:integer, yline=:chance)

plot_three_ind(fix4_value, "Rating of first minus\nsecond fixated item",
    "P(4th fixation is refixation\nto first fixated item)",
    (-6, 6), (0, 1),
    ; binning=:integer, xline=0)

plot_three_ind(fix3_value, "Rating of first fixated item",
    "P(third fixation is refixation\nto first fixated item)",
    (0, 10), (0, 1),
    ; binning=:integer,
    )


# %% ==================== Choce biases ====================
plot_both_ind(last_fix_bias, :last_rv, "P(last fixated item chosen)",
    (-6, 6), (0, 1)
    ; binning=:integer, xline=0, yline=:chance)

plot_both_ind(fixation_bias, "Final time advantage left [ms]", "P(left chosen)",
    (-2000, 2000), (0, 1)
    ; xline=0, yline=:chance,
    # trial_select=(t)->t.value[1] == 3
    )

plot_both_ind(first_fixation_duration, "First fixation duration [ms]", "P(first fixated chosen)",
    (0, 1500), (0, 1),
    yline= :chance )

plot_both_ind(first_fixation_duration_corrected, "First fixation duration [ms]", "corrected P(first fixated chosen)",
    (0, 1500), (-0.8, 0.8),
    yline=0 )

# %% --------
x, y = deserialize("tmp/bad_glm")
mean(y)




