# %%
using TypedTables
using SplitApplyCombine
using Statistics
using Serialization
include("utils.jl")

"This allows us to treat trials with the same value sets (but in different
order) as equivalent, which is useful for fitting"
function sort_value(t)
    rank = sortperm(t.value)
    remap = sortperm(rank)
    (
        value = t.value[rank],
        fixations = remap[t.fixations],
        choice = remap[t.choice],
        fix_times = t.fix_times,
    )
end

# if !@isdefined trials
#     trials = open(deserialize, "data/three_items.jls")
#     μ_emp, σ_emp = juxt(mean, std)(flatten(trials.value))
#     n_item = length(trials[1].value)
#     const rank_trials = map(sort_value, trials)
# end

load_dataset(num) = open(deserialize, "data/$(num)_items.jls")
function load_dataset(num, subject::Int)
    subject == -1 && return load_dataset(num)
    filter(load_dataset(num)) do t
        t.subject == subject
    end
end

function discretize_fixations(t; sample_time=100)
    mapmany(t.fixations, t.fix_times) do item, ft
        repeat([item], Int(round(ft/sample_time)))
    end
end

function train_test_split(trials, fold::String)
    fold == "all" && return (train=trials, test=trials)

    test_idx = if occursin("/", fold)
        this, total = parse.(Int, (split(fold, "/")))
        this:total:length(trials)
    else
        Dict(
            "odd" => 1:2:length(trials),
            "even" => 2:2:length(trials),
        )[fold]
    end
    train_idx = setdiff(eachindex(trials), test_idx)
    (train=trials[train_idx], test=trials[test_idx])
end

empirical_prior(trials) = juxt(mean, std)(flatten(trials.value))


