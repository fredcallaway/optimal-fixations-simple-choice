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
fit_mode = "joint"
fit_prior = "true"
out_path = "figs/$run_name/$fit_mode-$fit_prior"
mkpath(out_path)

both_sims = map(1:30) do i
    map(deserialize("results/$run_name/simulations/$fit_mode-$fit_prior/$i")) do sims
        reduce(vcat, sims)
    end
end |> invert;

NO_RIBBON = false
SKIP_BOOT = true
FAST = false

# %% ====================  ====================
out_path = "figs/optimization/"
mkpath(out_path)

both_sims = map(1:30) do i
    map(deserialize("results/investigate_optimization/simulations/$i")) do sims
        reduce(vcat, sims)
    end
end |> invert;

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

# plot_both(fixate_on_best, "Time since trial onset [ms]", "P(fixate best)",
#     (xticks=(0.5:2:8.5, string.(0:500:2000)), xlims=(0,8.5)),
#     binning=:integer, yline=:chance, align=:chance,
#     cutoff=2000, n_bin=8)
# FAST = true

plot_both(fixate_on_worst, "Time since trial onset [ms]", "P(fixate worst)",
    (xticks=(0.5:5:20.5, string.(0:500:2000)), xlims=(0,20.5)),
    binning=:integer, yline=:chance, align=:chance,
    cutoff=2000, n_bin=20)

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

plot_both(value_duration, "First fixated item rating",  "First fixation duration [ms]",
    binning=:integer, fix_select=firstfix)

# %% ====================  ====================
x, y = value_duration(load_dataset(3), fix_select=firstfix)
R"""
m = lm($y ~ $x)
summary(m)
"""
# %% ==================== Last fixations ====================

# plot_both(value_duration, "Item value",  "Fixation duration [ms]",
#     binning=:integer, fix_select=final)

plot_both(last_fixation_duration, "Chosen item time advantage\nbefore last fixation [ms]",
    # (xticks=[],),
    "Last fixation duration [ms]")

# %% ==================== Mechanism tests for 3 items ====================

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

plot_both(fixation_bias_corrected, "Final time advantage left [ms]", "corrected P(left chosen)",
    xline=0, yline=0)

plot_both(first_fixation_duration, "First fixation duration [ms]", "P(first fixated chosen)",
    yline= :chance )

plot_both(first_fixation_duration_corrected, "First fixation duration [ms]", "corrected P(first fixated chosen)",
    yline=0 )


#    =================================================
# %% ==================== SCRATCH ====================
#    =================================================

plot_threea("refixate_uncertain_alt", "Alternative fixation advantage\nof refixated item [ms]", "Density",
    (yticks = [],),
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials, ignore_current=true), 100., xmin=-1000, xmax=1000, line=(:black, 2)),
    plot_model=(sim; color=RED)->kdeplot!(refixate_uncertain(sim, ignore_current=true), 100., xmin=-1000, xmax=1000, line=(color, 2, 0.5)),
    xline=0
)

plot_three(fixate_by_uncertain, "Alternative fixation advantage\nof more-fixated item", "P(fixate more-fixated item)", yline=1/2,
    binning=Binning(-50:100:850))

# f2 = plot_three(fix4_uncertain,
#     "First minus second fixation duration",
#     "P(4th fixation is refixation\nto first-fixated item)",
#     # binning=Binning(-450:100:450),
#     xline=0, yline=0.5)


# %% ====================  ====================
FAST = true

function value_bias_alt(trials; fix_select=allfix)
    x = Float64[]; y = Float64[]
    for t in trials
        push!(x, relative_left(t.value))
        tft = total_fix_times(t; fix_select=fix_select)
        tft ./= (sum(tft) + eps())
        push!(y, tft[1])
    end
    x, y
end

plot_both(value_bias_alt, :left_rv, "Proportion fixate left";
    xline=0, yline=:chance)




# %% ====================  ====================

function correcting_times(trials; fix_select=allfix)
    x = zeros(6)
    n = zeros(6)
    for t in trials
        for (i, ft) in enumerate(t.fix_times)
            (i > 6 || i == length(t.fixations)) && break
            x[i] += ft
            n[i] += 1
        end
    end
    x ./ n
end

plot(correcting_times(both_trials[1]))
# %% ====================  ====================
FAST = false
function corrected_value_duration(trials)
    x = Float64[]; y = Float64[];
    ct = correcting_times(trials)
    for t in trials
        for i in eachindex(t.fixations)
            (i > 6 || i == length(t.fixations)) && break
            fi = t.fixations[i]; ti = t.fix_times[i]
            push!(x, relative(t.value)[fi])
            push!(y, ti - ct[i])
        end
    end
    x, y
end

plot_both(corrected_value_duration, "Relative item value",  "Corrected fixation duration [ms]",
    )

# %% ====================  ====================

function corrected_chosen_fix_time(trials)
    x = Bool[]; y = Float64[]
    ct = correcting_times(trials)
    for t in trials
        for i in eachindex(t.fixations)
            (i > 6 || i == length(t.fixations)) && break
            push!(x, t.fixations[i] == t.choice)
            push!(y, t.fix_times[i] - ct[i])
        end
    end
    x, y
end

plot_both(corrected_chosen_fix_time, "", "Corrected fixation duration [ms]",
    (xticks=(0:1, ["Unchosen", "Chosen"]), xlim=(-0.2, 1.2)),
    binning=:integer)


# %% ====================  ====================

function value_nfix(trials)
    n = n_item_(trials[1])
    x = Float64[]; y = Float64[];
    for t in trials
        push!(x, relative_left(t.value))
        push!(y, sum(t.fixations .== 1))
    end
    x, y
end
plot_both(value_nfix, :left_rv, "Number of fixations to left")

# %% ====================  ====================
FAST = false
function fix4_uncertain(trials)
    x = Float64[]; y = Float64[]; n = 3
    for t in trials
        if length(t.fixations) > n && sort(t.fixations[1:n]) == 1:n && unique_values(t)
            cft = total_fix_times(t; fix_select=(t,i)->i<=3)
            f3, f4 = t.fixations[3:4]
            if f3 == 1
                continue
            elseif f3 == 2
                alt = 3
            else
                alt = 2
            end
            push!(x, cft[1] - cft[alt])
            push!(y, f4 == 1)
        end
    end
    x, y
end


plot_three(fixate_by_uncertain, "Fixation advantage", "P(fixate)", yline=1/2, xline=0)

plot_three(fix4_uncertain,
    "Left item fixation advantage",
    "P(4th fixation is refixation\nto left item)",
    xline=0, yline=0.5)


# %% ====================  ====================
function uncertain_by_fixation(trials; ignore_current=false)
    n = n_item_(trials[1])
    @assert !(ignore_current && n == 2)
    options = Set(1:n)
    x = Int[]; y = Float64[]
    for t in trials
        cft = zeros(n)
        total = 0
        for i in eachindex(t.fixations)
            fix = t.fixations[i]
            fix_time = t.fix_times[i]
            if 3 < i < 7
                push!(x, i)
                prev = t.fixations[i-1]
                if ignore_current
                    other = pop!(setdiff(options, [prev, fix]))
                    push!(y, cft[fix] - cft[other])
                else
                    others = [i for i in options if i != fix]
                    push!(y, cft[fix] - mean(cft[others]))
                end
            end
            cft[fix] += fix_time
            total += fix_time
        end
    end
    return x, y
end
FAST = true

plot_both(uncertain_by_fixation, "Fixation number", "Fixation advantage",
    binning=:integer)

plot_three(uncertain_by_fixation, "Fixation number", "Fixation advantage",
    binning=:integer, ignore_current=true)



    # %% ====================  ====================

    function rank_duration(trials; fix_select=allfix)
        x = Float64[]; y = Float64[];
        for t in trials
            ranks = sortperm(sortperm(-t.value))
            for i in eachindex(t.fixations)
                fix_select(t, i) || continue
                fi = t.fixations[i]; ti = t.fix_times[i]
                push!(x, ranks[fi])
                push!(y, ti)
            end
        end
        x, y
    end

    plot_both(rank_duration, "Item value", "Fixation duration [ms]",
        # (xticks=(1:3, ["Unchosen", "Chosen"]),),
        binning=:integer, fix_select=nonfinal)

    # %% ====================  ====================
