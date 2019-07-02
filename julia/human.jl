# %%
using TypedTables
using SplitApplyCombine
using Statistics
using Serialization
include("utils.jl")

if !@isdefined trials
    const trials = open(deserialize, "human_trials.jls")
    const μ_emp, σ_emp = juxt(mean, std)(flatten(trials.value))
end

Trial = typeof(trials[1])

norm_value(t::Trial) = (t.value .- μ_emp) ./ σ_emp
function discretize_fixations(t; sample_time=100)
    mapmany(t.fixations, t.fix_times) do item, ft
        repeat([item], Int(round(ft/sample_time)))
    end
end
