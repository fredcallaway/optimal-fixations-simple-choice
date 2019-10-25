import CSV
using TypedTables
using SplitApplyCombine
# using Distributions
using StatsBase
using Lazy: @>>
using Serialization

include("utils.jl")

function process_data(reducer, csv, name)
    trials = @>> begin
        CSV.File(csv)
        Table
        group(x->(x.subject, x.trial))
        values
        map(reducer)
        Table
        # normalize_values!
    end
    serialize("data/$name.jls", trials)
    trials
end

process_data("../krajbich_PNAS_2011/data.csv", "three_items") do t
    r = t[1]
    (choice = argmax([r.choice1, r.choice2, r.choice3]),
     value = Int[r.rating1, r.rating2, r.rating3],
     subject = r.subject,
     trial = r.trial,
     rt = r.rt,
     fixations = combinedims([t.leftroi, t.middleroi, t.rightroi]) * [1, 2, 3],
     fix_times = t.eventduration)
end

process_data("../krajbich_NatNeu_2010/Data/fixations_final.csv", "two_items") do t
    r = t[1]
    (choice = Int(r.choice == 0 ? 2 : 1),
     value = Int[r.leftrating, r.rightrating],
     subject = Int(r.subject),
     trial = Int(r.trial),
     rt = Int(r.rt),
     fixations = Int.(t.roi),
     fix_times = Int.(t.event_duration))
end


