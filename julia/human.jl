# %%
using TypedTables
using SplitApplyCombine
using Statistics
using Serialization
include("utils.jl")

if !@isdefined trials
    const trials = open(deserialize, "data/three_items.jls")
    const μ_emp, σ_emp = juxt(mean, std)(flatten(trials.value))
    const n_item = length(trials[1].value)
end

Trial = typeof(trials[1])

norm_value(t::Trial) = (t.value .- μ_emp) ./ σ_emp
function discretize_fixations(t; sample_time=100)
    mapmany(t.fixations, t.fix_times) do item, ft
        repeat([item], Int(round(ft/sample_time)))
    end
end

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

const rank_trials = map(sort_value, trials)