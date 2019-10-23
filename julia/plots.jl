include("plots_base.jl")
include("params.jl")
include("box.jl")
using Glob
using StatsBase
plot([1,2])

# %% ====================  ====================
results = filter(get_results("new_pseudo")) do res
    exists(res, :reopt) &&
    length(load(res, :space)[:μ]) == 1
end

names = map(results) do res
    cv = load(res, :like_kws).index[1] == 1 ? "odd" : "even"
    run_name = join(["new_pseudo_fit_mu", cv, res.uuid], "_")
    # pretty(load(res, :reopt)[1])
end
results = results[sortperm(names)]
type2nt(p) = (;(v=>getfield(p, v) for v in fieldnames(typeof(p)))...)

map(results) do res
    mle = load(res, :mle)
    x = @Select(σ_obs, sample_cost, switch_cost, α)(mle)
    return x
    (x...,
     train_loss=mean(x[1] / x[3] for x in load(res, :reopt_like)),
     test_loss=mean(x[1] / x[3] for x in load(res, :test_like))
     )
end |> Table |> println

sims = asyncmap(policies) do pol
    simulate_experiment(pol; n_repeat=10)
end;

# %% ====================  ====================

mkpath("figs/$run_name")

function robplot(name, xlabel, ylabel, feature, bin_type=nothing, type=:line; after=()->0, kws...)
    hx, hy = feature(trials; kws...)
    bins = make_bins(bin_type, hx)
    f = plot(xlabel=xlabel, ylabel=ylabel)
    plot_human!(bins, hx, hy, type)
    for sim in sims
        mx, my = feature(sim; kws...)
        plot_model!(bins, mx, my, type, alpha=0.5)
    end
    after()
    savefig(f, "figs/$run_name/$name.pdf")
    display(f)
    f
end

function robplot(after::Function, name, xlabel, ylabel, feature, bin_type=nothing, type=:line; kws...)
    robplot(name, xlabel, ylabel, feature, bin_type, type; after=after, kws...)
end

function robplot(name; setup::Function, plot_human::Function, plot_model::Function, after=()->0)
    f = setup()
    plot_human(trials)
    for sim in sims
        plot_model(sim)
    end
    after()
    savefig(f, "figs/$run_name/$name.pdf")
    display(f)
    f
end


# %% ==================== Basic psychometrics ====================
include("features.jl")
left_rv = n_item == 2 ? "Left rating - right rating" : "Left rating - mean other rating"
best_rv = n_item == 2 ? "Best rating - worst rating" : "Best rating - mean other rating"

robplot("value_choice", left_rv, "P(left chosen)",
    value_choice, Binning(-4.5:1:4.5), after=cross013)

robplot("difference_time", best_rv, "Total fixation time [ms]",
    difference_time) do
        ylims!(0,7500)
    end

robplot("rt_kde";
    setup=()->plot(xlabel="Total fixation time", ylabel="Probability density"),
    plot_human=(trials)->kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=10000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=10000, line=(RED, 2, 0.5))
)

# %% ==================== Number of fixations ====================
robplot("n_fix_hist", "Number of fixations", "Proportion of trials",
    n_fix_hist, :integer, :discrete)

robplot("difference_nfix", "Maxium relative item value", "Number of fixations",
    difference_nfix, :integer)


# %% ==================== Fixation locations ====================

robplot("fixate_on_best", "Time since trial onset", "Probability of fixating\non highest-value item",
    fixate_on_best, :integer, cutoff=2000, n_bin=8) do
        xticks!(0.5:8.5, string.(0:250:2000))
        hline!([1/3], line=(:grey, 0.7), label="")
end

robplot("value_bias", "Relative item value", "Proportion fixation time",
    value_bias, after=cross013)

robplot("refixate_uncertain";
    setup=()->plot(xlabel="Fixation advantage of refixated item",
                   ylabel="Probability density",
                   title="Two items"),
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials), 100., xmin=-2000, xmax=2000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(refixate_uncertain(sim), 100., xmin=-2000, xmax=2000, line=(RED, 2, 0.5)),
    after=()->vline!([0], line=(:grey, 0.7), label="")
)

# %% ==================== Fixation durations ====================

robplot("binned_fixation_times", "Fixation type", "Fixation duration",
    binned_fixation_times, :integer, :discrete) do
        xticks!(1:4, ["first", "second", "middle", "last"])
end

robplot("full_fixation_times", "Fixation number", "Fixation duration",
    full_fixation_times, :integer)

robplot("chosen_fix_time_nonfinal", "", "Average fixation duration",
    chosen_fix_time, :integer, :discrete; fix_select=nonfinal) do
        xticks!(0:1, ["Unchosen", "Chosen"])
end

robplot("value_duration_first", "Item value", "Fixation duration", value_duration, :integer; fix_select=firstfix)

# %% ==================== Last fixations ====================

robplot("value_duration_final", "Item value", "Fixation duration", value_duration, :integer; fix_select=final)

robplot("last_fixation_duration", "Chosen item time advantage\nbefore last fixation", "Last fixation duration",
    last_fixation_duration)

# %% ==================== Mechanism tests for 3 items ====================
robplot("refixate_value", "Value of first vs. second fixated item", "Probability refixate first",
refixate_value)

robplot("refixate_4", "First minus second fixation duration", "Probability refixate first",
    refixate_4)

robplot("refixate_3_value", "Value of first-fixated item", "Probability third fixation\nto first fixated",
    refixate_3_value)


# %% ==================== Choice biases ====================

# %% ====================  ====================
cross013() = cross!(0, 1/n_item)

for sel in [allfix, nonfinal, final]
    # robplot("value_bias_$sel", "Relative item value", "Proportion fixation time",
    #     value_bias; fix_select=sel) do
    #         cross013()
    #         title!(string(sel))
    #     end



end

robplot("refixate_4", "First minus second fixation duration", "Probability refixate first",
    refixate_4)

robplot("refixate_total", "Total fixation time on first vs.\nsecond fixated item", "Probability refixate first",
    refixate_tft)

robplot("refixate_value", "Value of first vs. second fixated item", "Probability refixate first",
    refixate_value)

robplot("refixate_3_value", "Value of first-fixated item", "Probability third fixation\nto first fixated",
    refixate_3_value)
let
    for msplit in [:top, :bottom]
        robplot("refixate_3_$msplit", "First fixation duration", "Probability third fixation\nto first fixated",
            refixate_3, median_split=msplit)  do
                title!("$msplit half")
        end
    end
end

robplot("value_bias", "Relative item value", "Proportion fixation time",
    value_bias, after=cross013)

robplot("value_choice", "Relative item value", "Probability of choice",
    value_choice, :integer; after=cross013)

robplot("fixation_bias", "Relative fixation time", "Probability of choice",
    fixation_bias; after=cross013)

robplot("fixate_on_best", "Time since trial onset", "Probability of fixating\non highest-value item",
    fixate_on_best, :integer, cutoff=2000, n_bin=8) do
        xticks!(0.5:8.5, string.(0:250:2000))
        hline!([1/3], line=(:grey, 0.7), label="")
end

robplot("fixate_on_best_nonfinal", "Time since trial onset", "Probability of fixating\non highest-value item",
    fixate_on_best, :integer, cutoff=2000, n_bin=8, nonfinal=true) do
        xticks!(0.5:8.5, string.(0:250:2000))
        hline!([1/n_item], line=(:grey, 0.7), label="")
        title!("Nonfinal")
end
robplot("fixate_on_worst_nonfinal", "Time since trial onset", "Probability of fixating\non lowest-value item",
    fixate_on_worst, :integer, cutoff=2000, n_bin=8, nonfinal=true) do
        xticks!(0.5:8.5, string.(0:250:2000))
        hline!([1/3], line=(:grey, 0.7), label="")
        title!("Nonfinal")
end

robplot("fourth_rank", "Value rank of fourth-fixated item", "Proportion",
    fourth_rank, :integer, :discrete) do
        xticks!(1:3, ["best", "middle", "worst"])
end

robplot("first_fixation_duration", "Duration of first fixation", "Probability choose first fixated",
    first_fixation_duration)



robplot("binned_fixation_times", "Fixation type", "Fixation duration",
    binned_fixation_times, :integer, :discrete) do
        xticks!(1:4, ["first", "second", "middle", "last"])
end

robplot("last_fix_bias", "Last fixated item relative value", "Probability of choosing\nlast fixated item",
    last_fix_bias, :integer; after=cross013)
robplot("gaze_cascade", "Fixation number (aligned to choice)", "Proportion of fixations\nto chosen item",
    gaze_cascade, :integer)
# robplot("rt_kde", "Total fixation time", "Probability density", xlabel)

robplot("value_bias_chosen", "Relative item value", "Proportion fixation time",
    value_bias_split; chosen=true, after=cross013)
robplot("value_bias_unchosen", "Relative item value", "Proportion fixation time",
    value_bias_split; chosen=false, after=cross013)
# robplot("value_duration_", "Item value", "Fixation duration", value_duration_alt)
robplot("fixate_on_worst", "Time since trial onset", "Probability of fixating\non lowest-value item",
    fixate_on_worst, :integer, cutoff=2000, n_bin=8) do
         xticks!(0.5:8.5, string.(0:250:2000))
         hline!([1/3], line=(:grey, 0.7), label="")
end
robplot("fixation_bias_corrected", "Relative fixation time", "Corrected choice probability", fixation_bias_corrected,
        after=()->cross!(0,0))

robplot("refixate_uncertain";
    setup=()->plot(xlabel="Fixation advantage of refixated item",
                   ylabel="Probability density",
                   title="Two items"),
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials), 100., xmin=-2000, xmax=2000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(refixate_uncertain(sim), 100., xmin=-2000, xmax=2000, line=(RED, 2, 0.5)),
    after=()->vline!([0], line=(:grey, 0.7), label="")
)
for (name, sel) in pairs(selectors)
    robplot("value_duration_" * name, "Item value", "Fixation duration", value_duration_alt, :integer; selector=sel) do
        title!(name)
    end
end



 # %% ====================  ====================
function pretty(m::MetaMDP)
    println("Parameters")
    @printf "  σ_obs: %.2f\n  sample_cost: %.4f\n  switch_cost: %.4f\n" m.σ_obs m.sample_cost m.switch_cost
end
function pretty(pol::Policy)
    pretty(pol.m)
    @printf "  α: %.2f\n" pol.α
end
pretty(policies[1])

# %% ====================  ====================

function fixation_bias(trials)
    mapmany(trials) do t
        ft = total_fix_time(t)
        # invert((ft ./ sum(ft), t.choice .== 1:3))
        invert((ft .- mean(ft), t.choice .== 1:3))
    end |> Vector{Tuple{Float64, Bool}} |> invert
end


@time fixation_bias(sims[2]);
