import CSV
using TypedTables
using SplitApplyCombine
# using Distributions
using StatsBase
using Lazy: @>>
include("utils.jl")



function reduce_trial(t::Table)
    r = t[1]
    (choice = argmax([r.choice1, r.choice2, r.choice3]),
     value = Float64[r.rating1, r.rating2, r.rating3],
     subject = r.subject,
     trial = r.trial,
     rt = r.rt,
     fixations = combinedims([t.leftroi, t.middleroi, t.rightroi]) * [1, 2, 3],
     fix_times = t.eventduration)
end

function normalize_values!(trials)
    for (subj, g) in group(x->x.subject, trials)
        μ, σ = juxt(mean, std)(flatten(g.value))
        for v in g.value
            v .-= μ
            v ./= σ
        end
    end
    return trials
end

const trials = @>> begin
    CSV.File("../krajbich_PNAS_2011/data.csv")
    Table
    group(x->(x.subject, x.trial))
    values
    map(reduce_trial)
    Table
    # normalize_values!
end

using Serialization
open("human_trials.jls", "w") do f
    serialize(f, trials)
end
