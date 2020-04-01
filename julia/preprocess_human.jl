# Preprocesses CSV files into fast-loading julia serialized objects
# generates data/two_items.jls and data/three_items.jls
import CSV
using TypedTables
using SplitApplyCombine
# using Distributions
using StatsBase
using Serialization

include("utils.jl")

function process_data(reducer, csv, name)
    trials =
        CSV.File(csv) |>
        Table |>
        d -> group(x->(x.subject, x.trial), d) |>
        pairs |> Dict |> values |>
        d -> map(reducer, d) |>
        Table
        # normalize_values!
    serialize("data/$name.jls", trials)
    trials
end

process_data("../data/krajbich_PNAS_2011/data.csv", "three_items") do t
    t = Table(t)
    r = t[1]
    (choice = argmax([r.choice1, r.choice2, r.choice3]),
     value = Int[r.rating1, r.rating2, r.rating3],
     subject = r.subject,
     trial = r.trial,
     rt = r.rt,
     fixations = combinedims([t.leftroi, t.middleroi, t.rightroi]) * [1, 2, 3],
     fix_times = t.eventduration)
end

process_data("../data/krajbich_NatNeu_2010/data.csv", "two_items") do t
    t = Table(t)
    r = t[1]
    (choice = Int(r.choice == 0 ? 2 : 1),
     value = Int[r.leftrating, r.rightrating],
     subject = Int(r.subject),
     trial = Int(r.trial),
     rt = Int(r.rt),
     fixations = Int.(t.roi),
     fix_times = Int.(t.event_duration))
end


