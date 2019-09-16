include("plots_base.jl")

using Glob
using StatsBase

function get_pol(run_name)
    res = get_results(run_name)[end]
    load(res, :policy)
end

policies = get_pol.(["sep15_200_f1_alt", "sep15_200_f1", "sep15_200"])



function pretty(m::MetaMDP)
    println("Parameters")
    @printf "  σ_obs: %.2f\n  sample_cost: %.4f\n  switch_cost: %.4f\n" m.σ_obs m.sample_cost m.switch_cost
end
for pol in policies
    pretty(pol.m)
end


policies = map(get_results("sep15_200_f1_1_reopt")) do res
    try
        load(res, :policy)
    catch
        missing
    end
end |> skipmissing |> collect

# policies = open(deserialize, "tmp/soft_bmps_policies");

sims = asyncmap(policies; ntasks=5) do pol
    simulate_experiment(pol)
end

# %% ====================  ====================
function mean_reward(policy, n_roll, parallel)
    if parallel
        rr = @distributed (+) for i in 1:n_roll
            rollout(policy, max_steps=200).reward
        end
        return rr / n_roll
    else
        rr = mapreduce(+, 1:n_roll) do i
            rollout(policy, max_steps=200).reward
        end
        return rr / n_roll
    end

end

scores = map(policies) do pol
    mean_reward(pol, 100_000, true)
end
# %% ====================  ====================

run_name = "robustness_200_reopt"
mkdir("figs/$run_name")
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

function robplot(after::Function, name, xlabel, ylabel, feature, bin_type=nothing, type=:line, kws...)
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

# %% ====================  ====================
cross013() = cross!(0, 1/3)
robplot("value_choice", "Relative item value", "Probability of choice", value_choice, :integer; after=cross013)
robplot("fixation_bias", "Relative fixation time", "Probability of choice", fixation_bias; after=cross013)
robplot("value_bias", "Relative item value", "Proportion fixation time", value_bias; after=cross013)
robplot("fourth_rank", "Value rank of fourth-fixated item", "Proportion", fourth_rank, :integer, :discrete;
        after=()->xticks!(1:3, ["best", "middle", "worst"]))
robplot("first_fixation_duration", "Duration of first fixation", "Probability choose first fixated", first_fixation_duration)
robplot("last_fixation_duration", "Chosen item time advantage\nbefore last fixation", "Last fixation duration", last_fixation_duration)
robplot("difference_time", "Maximum relative item value", "Total fixation time", difference_time)
robplot("difference_nfix", "Maxium relative item value", "Number of fixations", difference_nfix, :integer)
robplot("binned_fixation_times", "Fixation type", "Fixation duration", binned_fixation_times, :integer, :discrete)
robplot("n_fix_hist", "Number of fixations", "Proportion of trials", n_fix_hist, :integer, :discrete)
robplot("last_fix_bias", "Last fixated item relative value", "Probability of choosing\nlast fixated item", last_fix_bias, :integer)
robplot("gaze_cascade", "Fixation number (aligned to choice)", "Proportion of fixations\nto chosen item", gaze_cascade, :integer)
# robplot("rt_kde", "Total fixation time", "Probability density", xlabel)
robplot("chosen_fix_time", "", "Average fixation duration", chosen_fix_time, :integer, :discrete;
        after=()->xticks!(0:1, ["Unchosen", "Chosen"]))
robplot("value_bias_chosen", "Relative item value", "Proportion fixation time", value_bias_split; chosen=true, after=cross013)
robplot("value_bias_unchosen", "Relative item value", "Proportion fixation time", value_bias_split; chosen=false, after=cross013)
# robplot("value_duration_", "Item value", "Fixation duration", value_duration_alt)
robplot("fixate_on_worst", "Time since trial onset", "Probability of fixating\non lowest-value item", fixate_on_worst,
         :integer, cutoff=2000, n_bin=8,
         after=()->(xticks!(0.5:8.5, string.(0:250:2000)); hline!([1/3], line=(:grey, 0.7), label="")))
robplot("fixation_bias_corrected", "Relative fixation time", "Corrected choice probability", fixation_bias_corrected,
        after=()->cross!(0,0))
robplot("full_fixation_times", "Fixation number", "Fixation duration", full_fixation_times, :integer)

robplot("rt_kde";
    setup=()->plot(xlabel="Total fixation time", ylabel="Probability density"),
    plot_human=(trials)->kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=5000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=5000, line=(RED, :dash, 2, 0.5))
)

robplot("refixate_uncertain";
    setup=()->plot(xlabel="Fixation advantage of refixated item",
                   ylabel="Probability density"),
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials), 100., line=(:black, 2)),
    plot_model=(sim)->kdeplot!(refixate_uncertain(sim), 100., line=(RED, 2, 0.5)),
    after=()->vline!([0], line=(:grey, 0.7), label="")
)

for (name, sel) in pairs(selectors)
    robplot("value_duration_" * name, "Item value", "Fixation duration", value_duration_alt, :integer; selector=sel)
end

# %% ====================  ====================

function fixation_bias(trials)
    mapmany(trials) do t
        ft = total_fix_time(t)
        # invert((ft ./ sum(ft), t.choice .== 1:3))
        invert((ft .- mean(ft), t.choice .== 1:3))
    end |> Vector{Tuple{Float64, Bool}} |> invert
end


@time fixation_bias(sims[2]);


# %% ====================  ====================

complot("fixate_on_best", "Time since trial onset", "Probability of fixating\non highest-value item") do sim
    plot_comparison(fixate_on_best, sim, :integer, cutoff=2000, n_bin=8)
end

complot("fixation_bias", "Relative fixation time", "Probability of choice") do sim
    plot_comparison(fixation_bias, sim)
end

complot("value_duration_middle", "Item value", "Fixation duration") do sim
    plot_comparison(value_duration_alt, sim, :integer; selector=selectors["middle"])
end

complot("split_fixations", "Fixation number", "Fixation duration") do sim
    f = plot()
    c = colormap("Reds", 8)
    for n in 2:8
        plot_model!(sim, x->fixation_times(x, n), :integer, :line, color=c[n],
        ylims=(0,6000))
    end
    f
end

# %% ====================  ====================
function complot(fun, name, xlabel, ylabel)
    figs = map(sortperm(-ers)) do i
        sim = sims[i]
        f = fun(sim)
        title!(@sprintf "%.3f" ers[i])
        display(f)
        f
    end
    f = plot(figs..., figsize=20, size=(800,600))
    xlabel!(figs[8], xlabel)
    ylabel!(figs[4], ylabel)
    savefig(f, "figs/robustness/$name.pdf")
end
