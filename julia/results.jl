using Dates
using Serialization

struct Results
    name::String
    timestamp::DateTime
    uuid::String
end
Base.isless(r1::Results, r2::Results) = isless(r1.timestamp, r2.timestamp)

function Results(name)
    r = Results(name, now(), string(rand(1:100000), base=62))
    mkpath(dir(r))
    println("Saving results to ", dir(r))
    save(r, :_r, r; verbose=false)
    r
end

function dir(r::Results)
    base = "results/$(r.name)/"
    stamp = replace(split(string(r.timestamp), ".")[1], ':' => '-')
    return base * stamp * "-" * r.uuid
end

get_result(path) = open(deserialize, "$path/_r")
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
load(r, name::String) = load(r, Symbol(name))

function exists(r::Results, name::Symbol)
    file = "$(dir(r))/$name"
    isfile(path(r, name))
end
