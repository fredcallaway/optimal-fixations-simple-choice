using Distributed
addprocs(40)

include("plots_base.jl")
include("pseudo_likelihood.jl")
include("params.jl")
include("box.jl")
using Glob
using StatsBase
plot([1,2])


# %% ====================  ====================
function load_results(dataset, name="feb17")
    if name == "OLD"
        name = (dataset == "both") ? "no-inner" : "test"
    end
    all_res = filter(get_results(name)) do res
        exists(res, :reopt) || return false
        length(load(res, :xy).y) >= 400 || return false
        args = load(res, :args)
        args["dataset"] == dataset || return false
        "n_inner" in keys(args) || return false
        # length(load(res, :xy).y) == 550 || return false
    end
end

function best_result(dataset)
    ress = load_results(dataset)
    ress[argmin(results_table(ress).train_loss)]
end

both_trials = load_dataset.(["two", "three"])


both_trials = map(["two", "three"]) do num
    load_dataset(num)[1:2:end]
end;
# dir(best_result("both"))

# %% ==================== Summarize fits ====================
res = load_results("both")[1]
param_names = [free(load(res, :outer_space)); free(load(res, :inner_space))]

function load_x(res)
    args = load(res, :args)
    mle = load(res, :mle)
    os = load(res, :outer_space)
    is = load(res, :inner_space)
    t = type2dict(mle)
    vcat(os(t), is(t))
end

mkpath("figs/fits/")
function plot_fits(results, name; kws...)
    f = plot(
        xticks=(1:5, string.(param_names)),
        ylabel="Normalized parameter value",
        title=name)
    for res in results
        x = load_x(res)
        # plot!(x.x, color=lab[x.fold], lw=2)
        plot!(x, lw=2; ylim=(0,1), kws...)
    end
    savefig("figs/fits/$name.pdf")
    display(f)
end

open("old_fits.txt", "w") do f
    for dataset in ["two", "three", "both"]
        all_res = load_results(dataset, "OLD")
        println(f, "\n\n********* $dataset *********",)
        plot_fits(all_res, dataset)
        show(f, results_table(all_res), allcols=true, splitcols=false)
    end
end;

# %% ==================== Separate fitting results ====================
ress = best_result.(["two", "three"])
run_name = "separate_fit_nov25"

both_sims = map(1:2) do i
    res = ress[i]
    trials = both_trials[i]
    ds = load(res, :datasets)[1]
    prm = load(res, :mle)
    policies = load(res, :reopt)[1].policies
    asyncmap(policies; ntasks=2) do pol
        @assert pol.α == prm.α
        simulate_experiment(pol, trials; μ=prm.β_μ * ds.μ_emp, σ=prm.β_σ * ds.σ_emp,
            sample_time=prm.sample_time, n_repeat=10)
    end
end

# %% ==================== Joint fitting results ====================
# run_name = "joint_fit_dec3"
# run_name = "joint_feb17"
run_name = "old_check"

function get_sims(res; load_previous=true)
    load_previous && exists(res, :both_sims) && return load(res, :both_sims)
    reopt = load(res, :reopt);
    prm = load(res, :mle)
    datasets = load(res, :datasets);
    both_sims = map(1:2) do i
        trials = both_trials[i]
        μ_emp, σ_emp = datasets[i].μ_emp, datasets[i].σ_emp
        policies = reopt[i].policies
        asyncmap(policies) do pol
            simulate_experiment(pol, trials; μ=prm.β_μ * μ_emp, σ=prm.β_σ * σ_emp,
                sample_time=prm.sample_time, n_repeat=10)
        end
    end
    save(res, :both_sims, both_sims)
    return both_sims
end

res = best_result("both")
@time both_sims = get_sims(res);
# multi_sims = map(load_recent_results("both")) do res
#     get_sims(res, load_previous=false);
# end;

# %% ====================  ====================
run_name = "all_sobol2"
@time both_sims = map(1:30) do i
    map(deserialize("results/sobol/sims/$i")) do sims
        reduce(vcat, sims)
    end
end |> invert;

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
    # plot_human!(bins, hx, hy, type)

    if FAST
        sims = sims[1:2]
    end
    sims = reverse(sims)
    colors = range(colorant"red", stop=colorant"blue",length=length(sims))
    for (c, sim) in zip(colors, sims)
        mx, my = feature(sim; kws...)
        plot_model!(bins, mx, my, type, alpha=0.2, color=c)
    end
    plot_human!(bins, hx, hy, type)

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

    # plot_human(trials)

    if FAST
        sims = sims[1:4]
    end
    sims = reverse(sims)
    colors = range(colorant"red", stop=colorant"blue",length=length(sims))
    for (c, sim) in zip(colors, sims)
        plot_model(sim, color=c)
    end
    plot_human(trials)

    make_lines!(xline, yline, trials)
    f
end



# left_rv = n_item == 2 ? "Left rating - right rating" : "Left rating - mean other rating"
# best_rv = n_item == 2 ? "Best rating - worst rating" : "Best rating - mean other rating"
DISABLE_ALIGN = true
function plot_both(feature, xlab, ylab, plot_kws=(); yticks=true, align=:default, name=string(feature), kws...)
    xlab1, xlab2 =
        (xlab == :left_rv) ? ("Left rating - right rating", "Left rating - mean other rating") :
        (xlab == :best_rv) ? ("Best rating - worst rating", "Best rating - mean other rating") :
        (xlab == :last_rv) ? ("Last fixated rating - other rating", "Last fixated rating - mean other") :
        (xlab, xlab)

    # ff = plot_one(feature, xlab, ylab, trials, sims, plot_kws; kws...)
    # if haskey(Dict(kws), :fix_select)
    #     name *= "_$(kws[:fix_select])"
    # end
    # savefig(ff, "figs/$run_name/$name.pdf")
    # return ff
    if !yticks
        plot_kws = (plot_kws..., yticks=[])
        ylab *= "\n"
    end

    f1 = plot_one(feature, xlab1, ylab, both_trials[1], both_sims[1], plot_kws; kws...)
    f2 = plot_one(feature, xlab2, ylab, both_trials[2], both_sims[2], plot_kws; kws...)

    ylabel!(f2, yticks ? "  " : " \n ")
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
    # return ff
    display(ff)
    return
end

include("features.jl")
include("plots_base.jl")
NO_RIBBON = true
SKIP_BOOT = true
# %% ==================== Basic psychometrics ====================
# run_name = "no_inner"
mkpath("figs/$run_name")

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

plot_both(fixate_on_worst, "Time since trial onset [ms]", "P(fixate best)",
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



# %% ====================  ====================

function foo(trials; trial_select=(t)->true)
    x = Float64[]; y = Bool[];
    for t in trials
        trial_select(t) || continue
        push!(x, relative_left(total_fix_times(t; fix_select=nonfinal)))
        push!(y, t.choice == 1)
    end
    x, y
end

plot_both(foo, "Final time advantage left [ms]", "P(left chosen)",
; xline=0, yline=:chance,
# trial_select=(t)->t.value[1] == 3
)

# %% ====================  ====================
FAST = true
