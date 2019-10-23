include("plots_base.jl")
include("box.jl")
using Glob
using StatsBase
plot([1,2])

# %% ====================  ====================
function get_pol(run_name)
    res = get_results(run_name)[end]
    load(res, :policy)
end

policies = get_pol.(["sep15_200_f1_alt", "sep15_200_f1", "sep15_200"])

# %% ====================  ====================
run_name = "sep16_mu"
policies = map(get_results("$(run_name)_1_reopt")) do res
    try
        load(res, :policy)
    catch
        missing
    end
end |> skipmissing |> collect
prm = load(get_results(run_name)[1], :params)
# %% ====================  ====================
# run_name = "fit_pseudo_reopt"
run_name = "fit_pseudo_3_reopt"
include("gp_min.jl")

policies = map(get_results(run_name)) do res
    load(res, :policy)[1]
end

include("params.jl")
prm = open(deserialize, "tmp/fit_pseudo_3_model_mle")
@assert policies[1].m.σ_obs ≈ prm.σ_obs
@assert policies[1].α ≈ prm.α
pretty(policies[1])
# %% ====================  ====================
# run_name = "fit_pseudo_reopt"
run_name = "fit_pseudo_4_reopt"
policies = map(get_results(run_name)) do res
    load(res, :policy)
end
# %% ====================  ====================
run_name = "pseudo_3"
res = get_results(run_name)[3]
policies = load(res, :reopt)
policies[1] |> pretty
# %% ====================  ====================
run_name = "pseudo_3_epsilon"
res = get_results(run_name)[1]
policies = load(res, :reopt)

# %% ====================  ====================
run_name = "pseudo_4_top"
res = get_results(run_name)[3]
policies = load(res, :reopt)
load(res, :metrics)
policies[1] |> pretty

# %% ====================  ====================
run_name = "eps_0.4"
res = get_result("results/pseudo_mu_cv/2019-10-11T14-54-09-Btf/")
policies = load(res, :reopt)
policies[1] |> pretty

# %% ====================  ====================
results = filter(get_results("pseudo_mu_cv")) do res
    exists(res, :reopt) &&
    n_free(load(res, :space)) == 4
end
i = 1
policies = load(results[i], :reopt)
run_name = "pseudo_mu_cv-$i"
# %% ====================  ====================
run_name = "propfix"
res = get_result("results/fit_pseudo_preopt/2019-10-13T10-37-07-KEO")
policies = load(res, :reopt)
policies[1] |> pretty

# %% ====================  ====================
run_name = "choice_only"
res = get_result("results/fit_pseudo_preopt/2019-10-14T11-20-42-C8W/")
policies = load(res, :reopt)
policies[1] |> pretty

# %% ====================  ====================
run_name = "fixed"
res = get_result("results/fit_pseudo_preopt/2019-10-14T11-43-42-IoI/")
policies = load(res, :reopt)
load(res, :metrics)
policies[1] |> pretty

# %% ====================  ====================
run_name = "top_fix_prop"
res = get_result("results/fit_pseudo_preopt/2019-10-14T12-31-13-5UF/")
policies = load(res, :reopt)



# %% ====================  ====================
results = filter(get_results("new_pseudo")) do res
    exists(res, :reopt) &&
    length(load(res, :space)[:μ]) == 1
end


display("")
include("box.jl")
include("params.jl")
names = map(results) do res
    cv = load(res, :like_kws).index[1] == 1 ? "odd" : "even"
    run_name = join(["new_pseudo_fit_mu", cv, res.uuid], "_")
    # pretty(load(res, :reopt)[1])
end

results = results[sortperm(names)]

# %% ====================  ====================
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


# %% ====================  ====================

include("robust_helper.jl")
for res in results
    policies = load(res, :reopt)
    cv = load(res, :like_kws).index[1] == 1 ? "odd" : "even"
    run_name = join(["new_pseudo_fit_mu", cv, res.uuid], "_")
    make_plots(policies, run_name)
end

# %% ==================== GENERATE SIMS ====================
res = results[1]
policies = load(res, :reopt)
cv = load(res, :like_kws).index[1] == 1 ? "odd" : "even"
run_name = join(["new_pseudo_fit_mu", cv, res.uuid], "_")

# %% ====================  ====================
res = get_result("results/two_items/2019-10-21T17-22-29-2lG/")
run_name = "two_item_corrected"
res = get_result("results/two_items/2019-10-21T23-54-52-9lg")
policies = load(res, :reopt)

pretty(policies[1])

# %% ====================  ====================
# include("bmps_moments_fitting.jl")
sims = asyncmap(policies) do pol
    simulate_experiment(pol; n_repeat=10)
    # simulate_experiment(pol, prm.μ, prm.σ; n_repeat=10)
end;
# describe_vec(sim_loss.(sims))

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

function plot_model!(bins, x, y, type=:line; kws...)
    vals = bin_by(bins, x, y)
    if type == :line
        plot!(mids(bins), estimator.(vals),
              ribbon=ci_err.(estimator, vals),
              fillalpha=0.1,
              color=RED,
              line=(RED, 1),
              label="";
              kws...)
    elseif type == :discrete
        plot!(mids(bins), estimator.(vals),
              # yerr=ci_err.(estimator, vals),
              grid=:none,
              line=(RED, 1),
              marker=(7, :diamond, RED, stroke(0)),
              label="";
              kws...)
    else
        error("Bad plot type : $type")
    end
end

# %% ====================  ====================
robplot("difference_time", "Maximum relative item value", "Total fixation time",
    difference_time)


robplot("rt_kde";
    setup=()->plot(xlabel="Total fixation time", ylabel="Probability density"),
    plot_human=(trials)->kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=10000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=10000, line=(RED, 2, 0.5))
)
# %% ====================  ====================
plot()
mrv = map(trials.value) do t
    maximum(t .- mean(t))
end
histogram(mrv)
# %% ====================  ====================

feature = difference_time
hx, hy = feature(trials)
bins = make_bins(nothing, hx)

unique(collect(skipmissing(bins.(hx))))
mys = map(sims) do sim
    x, y = feature(sim)
    vals = bin_by(bins, feature(sim)...)
end
i = argmax([mean(m[1]) for m in mys])
sim = sims[i]
y = mys[i]
y = y[1:6]

mx, my = feature(sims[i])
mean(my)
mean(sum.(sim.fix_times))
mean(sum.(trials.fix_times))
mean.(y)

mhy = mean.(bin_by(bins, hx, hy))[1:6]

w = counts(collect(skipmissing(bins.(hx))))
w = w ./ sum(w)

w' * mean.(y)
w
round.(Int, mean.(y) .- mhy)

# %% ====================  ====================
function refixate_uncertain(trials)
    options = Set(1:n_item)
    x = Float64[]
    for t in trials
        cft = zeros(n_item)
        total = 0
        for i in eachindex(t.fixations)
            fix = t.fixations[i]
            fix_time = t.fix_times[i]
            if i > 2
                prev = t.fixations[i-1]
                alt = n_item == 2 ? prev : pop!(setdiff(options, [prev, fix]))
                push!(x, cft[fix] - cft[alt])
            end
            cft[fix] += fix_time
            total += fix_time
        end
    end
    return x
end

# %% ====================  ====================
robplot("refixate_uncertain";
    setup=()->plot(xlabel="Fixation advantage of refixated item",
                   ylabel="Probability density",
                   title="Two items"),
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials), 100., xmin=-2000, xmax=2000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(refixate_uncertain(sim), 100., xmin=-2000, xmax=2000, line=(RED, 2, 0.5)),
    after=()->vline!([0], line=(:grey, 0.7), label="")
)

# %% ====================  ====================
cross013() = cross!(0, 1/n_item)

for sel in [allfix, nonfinal, final]
    # robplot("value_bias_$sel", "Relative item value", "Proportion fixation time",
    #     value_bias; fix_select=sel) do
    #         cross013()
    #         title!(string(sel))
    #     end

    robplot("chosen_fix_time_$sel", "", "Average fixation duration",
        chosen_fix_time, :integer, :discrete; fix_select=sel) do
            xticks!(0:1, ["Unchosen", "Chosen"])
            title!(string(sel) * " $n_item items")
    end

    # robplot("full_fixation_times_$sel", "Fixation number", "Fixation duration",
    #     full_fixation_times, :integer; fix_select=sel) do
    #         title!(string(sel) * " $n_item items")
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

robplot("last_fixation_duration", "Chosen item time advantage\nbefore last fixation", "Last fixation duration",
    last_fixation_duration)

robplot("difference_time", "Maximum relative item value", "Total fixation time",
    difference_time)

robplot("difference_nfix", "Maxium relative item value", "Number of fixations",
    difference_nfix, :integer)

robplot("binned_fixation_times", "Fixation type", "Fixation duration",
    binned_fixation_times, :integer, :discrete) do
        xticks!(1:4, ["first", "second", "middle", "last"])
end

robplot("n_fix_hist", "Number of fixations", "Proportion of trials",
    n_fix_hist, :integer, :discrete)
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

robplot("rt_kde";
    setup=()->plot(xlabel="Total fixation time", ylabel="Probability density"),
    plot_human=(trials)->kdeplot!(sum.(trials.fix_times), 300., xmin=0, xmax=5000, line=(:black, 2)),
    plot_model=(sim)->kdeplot!(sum.(sim.fix_times), 300., xmin=0, xmax=5000, line=(RED, 2, 0.5))
)

robplot("refixate_uncertain";
    setup=()->plot(xlabel="Fixation advantage of refixated item",
                   ylabel="Probability density"),
    plot_human=(trials)->kdeplot!(refixate_uncertain(trials), 100., line=(:black, 2)),
    plot_model=(sim)->kdeplot!(refixate_uncertain(sim), 100., line=(RED, 2, 0.5)),
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
