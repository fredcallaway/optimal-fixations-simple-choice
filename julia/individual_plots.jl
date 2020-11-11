INTERACTIVE = basename(PROGRAM_FILE) == ""

include("plots_base.jl")
PARAMS = [
    (run_name="revision", dataset="joint", prior="fit", color=(colorant"#ff6167", colorant"#b00007"), alpha=1, n_sim=30),
    # (run_name="revision", dataset="joint", prior="zero", color=(colorant"#699efa", colorant"#003085"), alpha=0.1, n_sim=30),
    # (run_name="addm", color=(colorant"#4DCE3F", colorant"#3B9B31"), alpha=1, n_sim=1),
]
out_path = "figs/revision/individual"

rm(out_path, recursive=true, force=true)
mkpath(out_path)


function plot_individuals(feature, xlab, ylab, plot_kws=(); three_only=false, n_col=5, kws...) 
    # kws = (kws..., plot_model=false)
    for n_item in (three_only ? [3] : [2, 3])
        xlab = fmt_xlab(xlab, n_item)
        subjects = both_trials[n_item - 1].subject |> unique

        S = both_trials[n_item - 1].subject
        sc = countmap(S)
        subjects = filter(s->sc[s] == 50, unique(S))
        subjects = subjects[1:min(length(subjects), 30)] |> sort

        # push!(subjects, subjects[end], subjects[end])

        plots = map(subjects) do subj
            plot_one(feature, n_item, xlab, ylab, (plot_kws..., title=subj); 
                subject=subj, kws...)
        end
        
        n_row = cld(length(plots), n_col)

        empty_plot = plot(axis=false, grid=false)
        while length(plots) < n_row * n_col
            push!(plots, deepcopy(empty_plot))
        end

        pp = permutedims(reshape(plots, n_col, n_row))

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

        ff = plot(collect(permutedims(pp))..., layout=(n_row,n_col), size=(n_col*200, n_row*200))

        name = string(feature)
        savefig(ff, "$out_path/$name-$n_item.pdf")
    end
end

# %% --------
plot_individuals(value_bias, :left_rv, "Proportion fixate left",
    (ylim=(-0.05, 1.05), xlim=(-6, 6)); binning=Binning(-6:4:6),
    xline=0, yline=:chance)

# %% --------
plot_individuals(fix4_value, "Rating of first minus\nsecond fixated item",
    "P(4th fixation is refixation\nto first fixated item)",
    (xlim=(-6, 6), ylim=(-0.05, 1.05)); binning=Binning(-6:4:6),
    three_only=true, xline=0)

# %% --------
plot_individuals(value_duration, "First fixated item rating",  "First fixation duration [ms]",
    (xlim=(0, 9), ylim=(0, 1000));
    binning=Binning(0:3:9), fix_select=firstfix)

# %% --------
plot_individuals(value_choice, :left_rv, "P(left chosen)",
    (xlim=(-6, 6), ylim=(-0.05, 1.05)); binning=Binning(-6:4:6),
    xline=0, yline=:chance)

# %% --------
plot_individuals(binned_fixation_times, "Fixation number", "Fixation duration [ms]",
    (ylim=(0, 1300), xticks=(1:7, ["1", "2", "3", "4", "5", ">5", "F"]),),
    binning=:integer)

# %% --------

plot_individuals(first_fixation_duration, "First fixation duration [ms]", "P(first fixated chosen)",
    yline= :chance )

# %% --------
plot_individuals(nfix_hist, "Number of fixations", "Proportion of trials",
    (xticks=[1,5,10], ),
    binning=:integer, type=:discrete)

# %% --------

plot_individuals(plot_uncertain_alt, "Alternative fixation advantage\nof fixated item [ms]", "Density",
    xline=0, yticks = false, name="refixate_uncertain_alt", manual=true, three_only=true
)
# %% --------

plot_individuals(last_fix_bias, :last_rv, "P(last fixated item chosen)",
    binning=Binning(-6:4:6), xline=0, yline=:chance)

# %% --------
plot_individuals(fixation_bias, "Final time advantage left [ms]", "P(left chosen)",
    binning=Binning(-1500:1500:1500); xline=0, yline=:chance,
    # trial_select=(t)->t.value[1] == 3
    )

# %% --------

gr()
f1 = plot([1,2])
f2 = plot([1,3])
plot(f1, f2, f1)
