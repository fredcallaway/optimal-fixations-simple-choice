include("plots_base.jl")


# %% ====================  ====================
include("results.jl")
@with_kw mutable struct Params
    α::Float64
    σ_obs::Float64
    sample_cost::Float64
    switch_cost::Float64
    µ::Float64
    σ::Float64
    sample_time::Float64
end
Params(d::AbstractDict) = Params(;d...)

MetaMDP(prm::Params) = MetaMDP(
    3,
    prm.σ_obs,
    prm.sample_cost,
    prm.switch_cost,
)

run_name = "fit_pseudo_3"
policies, likes = map(get_results(run_name)) do res
    out = load(res, :out)
    out.policy, out.likelihood
end |> invert

maximum(likes)
policy = policies[argmax(likes)]

# sim = simulate_experiment(policy; n_repeat=10)

open("tmp/fit_pseudo_3_policy", "w+") do f
    serialize(f, policy)
end

# maximum(likes) / BASELINE
# %% ====================  ====================
run_name = "fit_pseudo_random"
rando = RandomPolicy(m)
sim = simulate_experiment(rando)

# %% ====================  ====================
run_name = "aug21"
best = open(deserialize, "results/aug21/2019-08-21T23-17-54-EgG/best")
sim = best.sim
# %% ====================  ====================
run_name = "original"
sim = open(deserialize, "tmp/original_sim")

# %% ====================  ====================
run_name = "new"
sim = open(deserialize, "tmp/new_sim")

# %% ====================  ====================
run_name = "sep13"
policy = open(deserialize, "tmp/sep13_pol")
sim = simulate_experiment(policy)

# %% ====================  ====================
run_name = "sep14_200"
policy = open(deserialize, "tmp/sep14_200_pol")
sim = simulate_experiment(policy)

# %% ====================  ====================
res = "2019-08-23T02-47-19-7HI"
run_name = "robustness"
policy = open(deserialize, "results/robustness-N/$res/optimal")

# %% ====================  ====================
run_name = "sep15_200_f1_alt"
res = get_results(run_name)[end]
policy = load(res, :policy)
sim = simulate_experiment(policy)

# %% ====================  ====================
run_name = "sep18_fit_pseudo_3"
policy = open(deserialize, "tmp/fit_pseudo_3_policy_100").policy

# %% ====================  ====================
run_name = "sep18_pseudo_4"
policy = open(deserialize, "tmp/fit_pseudo_4_model_mle").policy
sim = simulate_experiment(policy; n_repeat=10)
# %% ====================  ====================
mkpath("figs/$run_name")

fig("value_choice") do
    plot_comparison(value_choice, sim, :integer)
    cross!(0, 1/3)
    xlabel!("Relative item value")
    ylabel!("Probability of choice")
end

fig("fixation_bias") do
    plot_comparison(fixation_bias, sim)
    cross!(0, 1/3)
    xlabel!("Relative fixation time")
    ylabel!("Probability of choice")
end

fig("value_bias") do
    plot_comparison(value_bias, sim)
    cross!(0, 1/3)
    xlabel!("Relative item value")
    ylabel!("Proportion fixation time")
end

fig("fixate_on_best") do
    plot_comparison(fixate_on_best, sim, :integer, cutoff=2000, n_bin=8)
    xticks!(0.5:8.5, string.(0:250:2000))
    hline!([1/3], line=(:grey, 0.7), label="")
    xlabel!("Time since trial onset")
    ylabel!("Probability of fixating\non highest-value item")
end

fig("fourth_rank") do
    plot_comparison(fourth_rank, sim, :integer, :discrete)
    xlabel!("Value rank of fourth-fixated item")
    ylabel!("Proportion")
    xticks!(1:3, ["best", "middle", "worst"])
end

fig("first_fixation_duration") do
    plot_comparison(first_fixation_duration, sim, )
    xlabel!("Duration of first fixation")
    ylabel!("Probability choose first fixated")
end

fig("last_fixation_duration") do
    plot_comparison(last_fixation_duration, sim)
    xlabel!("Chosen item time advantage\nbefore last fixation")
    ylabel!("Last fixation duration")
end

fig("difference_time") do
    plot_comparison(difference_time, sim)
    xlabel!("Maximum relative item value")
    ylabel!("Total fixation time")
end

fig("difference_nfix") do
    plot_comparison(difference_nfix, sim, :integer)
    xlabel!("Maxium relative item value")
    ylabel!("Number of fixations")
end

fig("binned_fixation_times") do
    plot_comparison(binned_fixation_times, sim, :integer, :discrete)
    xticks!(1:4, ["first", "second", "middle", "last"])
    xlabel!("Fixation type")
    ylabel!("Fixation duration")
end

fig("n_fix_hist") do
    plot_comparison(n_fix_hist, sim, :integer, :discrete)
    xlabel!("Number of fixations")
    ylabel!("Proportion of trials")
end

fig("last_fix_bias") do
    plot_comparison(last_fix_bias, sim, :integer)
    cross!(0, 1/3)
    xlabel!("Last fixated item relative value")
    ylabel!("Probability of choosing\nlast fixated item")
end

fig("gaze_cascade") do
    plot_comparison(gaze_cascade, sim, :integer)
    xlabel!("Fixation number (aligned to choice)")
    ylabel!("Proportion of fixations\nto chosen item")
end

fig("rt_kde") do
    plot(xlabel="Total fixation time", ylabel="Probability density")
    kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=5000, line=(:black, 2), )
    kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=5000, line=(RED, :dash, 2), )
end

fig("chosen_fix_time") do
    plot_comparison(chosen_fix_time, sim, :integer, :discrete)
    xticks!(0:1, ["Unchosen", "Chosen"])
    ylabel!("Average fixation duration")
end

fig("value_bias_chosen") do
    plot_comparison(value_bias_split, sim; chosen=true)
    cross!(0, 1/3)
    xlabel!("Relative item value")
    ylabel!("Proportion fixation time")
end

fig("value_bias_unchosen") do
    plot_comparison(value_bias_split, sim; chosen=false)
    cross!(0, 1/3)
    xlabel!("Relative item value")
    ylabel!("Proportion fixation time")
end

for (name, sel) in pairs(selectors)
    fig("value_duration_" * name) do
        plot_comparison(value_duration_alt, sim, :integer; selector=sel)
        xlabel!("Item value")
        ylabel!("Fixation duration")
    end
end

fig("fixate_on_worst") do
    plot_comparison(fixate_on_worst, sim, :integer, cutoff=2000, n_bin=8)
    xticks!(0.5:8.5, string.(0:250:2000))
    hline!([1/3], line=(:grey, 0.7), label="")
    xlabel!("Time since trial onset")
    ylabel!("Probability of fixating\non lowest-value item")
end

fig("fixation_bias_corrected") do
    plot_comparison(fixation_bias_corrected, sim)
    cross!(0, 0)
    xlabel!("Relative fixation time")
    ylabel!("Corrected choice probability")
end

fig("full_fixation_times") do
    plot_comparison(full_fixation_times, sim, :integer)
    xlabel!("Fixation number")
    ylabel!("Fixation duration")
end

fig("refixate_uncertain") do
    plot(xlabel="Fixation advantage of refixated item",
        ylabel="Probability density")
    kdeplot!(refixate_uncertain(trials), 100., line=(:black, 2))
    kdeplot!(refixate_uncertain(sim), 100., line=(RED, 2))
    vline!([0], line=(:grey, 0.7), label="")
end

fig("split_fixations") do
    plot()
    c = colormap("Blues", 8)
    for n in 2:8
        plot_human!(x->fixation_times(x, n), :integer, :line, color=c[n])
    end
    xlabel!("Fixation number")
    ylabel!("Fixation duration")
end

fig("split_fixations_model") do
    plot()
    c = colormap("Reds", 8)
    for n in 2:8
        plot_model!(sim, x->fixation_times(x, n), :integer, :line, color=c[n])
    end
    xlabel!("Fixation number")
    ylabel!("Fixation duration")
end

# %% ====================  ====================
function rev_fixation_times(trials)
    x = Tuple{Int, Float64}[]
    for t in trials
        f = t.fix_times
        length(f) < 4 && continue
        for i in 0:min(4, length(f)-1)
            push!(x, (-i, f[end-i]))
        end
    end
    invert(x)
end

function rev_fixation_times(trials, n)
    x = Tuple{Int, Float64}[]
    for t in trials
        f = t.fix_times
        length(f) != n && continue
        for i in 0:n-1
            push!(x, (-i, f[end-i]))
        end
    end
    invert(x)
end

fig("split_fixations_rev") do
    plot()
    c = colormap("Blues", 8)
    for n in 2:8
        plot_human!(x->rev_fixation_times(x, n), :integer, :line, color=c[n])
    end
    xlabel!("Fixation number (from final)")
    ylabel!("Fixation duration")
end

fig("split_fixations_model_rev") do
    plot()
    c = colormap("Reds", 8)
    for n in 2:8
        plot_model!(x->rev_fixation_times(x, n), :integer, :line, color=c[n])
    end
    xlabel!("Fixation number (from final)")
    ylabel!("Fixation duration")
end

fig("rev_fixation_times") do
    plot_comparison(rev_fixation_times, sim, :integer)
    xlabel!("Fixation number (from final)")
    ylabel!("Fixation duration")
end
