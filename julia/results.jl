using Dates
using Serialization

struct Results
    name::String
    timestamp::DateTime
end
Base.isless(r1::Results, r2::Results) = isless(r1.timestamp, r2.timestamp)

function Results(name)
    r = Results(name, now())
    mkpath(dir(r))
    println("Saving results to ", dir(r))
    save(r, :_r, r; verbose=false)
    r
end

dir(r::Results) = "results/$(r.name)/" * replace(split(string(r.timestamp), ".")[1], ':' => '-')
get_result(name, timestamp) = open(deserialize, "results/$name/$timestamp/_r")
get_results(name) = sort!(get_result.(name, readdir("results/$name")))

path(r::Results, name::Symbol) = "$(dir(r))/$name"

function save(r::Results, name::Symbol, value; verbose=true)
    file = path(r, name)
    open(file, "w") do f
        serialize(f, value)
    end
    verbose && println("Wrote $file")
end

function load(r::Results, name::Symbol)
    file = "$(dir(r))/$name"
    open(deserialize, path(r, name))
end
