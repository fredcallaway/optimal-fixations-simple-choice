import JSON
using UUIDs: uuid4
using Dates: now

function save(job::Job, name::Symbol, value)
    d = Dict(
        :job => job,
        :time => now(),
        :value => value
    )
    open(result_file(job, name), "w") do f
        write(f, JSON.json(d))
    end
    println("Wrote $(result_file(job, name))")
end
load(job::Job, name::Symbol) = JSON.parsefile(result_file(job, name))[string(name)]

function save(group, table, entry)
    id = uuid4()
    path = "runs/$group/results/$table"
    println(path)
    mkpath(path)
    file = "$path/$id.json"
    d[:write_time] = now()
    open(file, "w") do f
        write(f, JSON.json(d))
    end
    println(path)
    JLD.save(file, "entry", entry)
end

function save(g, t, entry::NamedTuple)

end

entry = (x=1, y=2)

Dict(entry)

JLD.save("test.jld", "foo", )
JLD.save("test.jld", Dict("foo" => (x=1, y=2)))

save("foo", "bar", (x=1, y=2))

using Glob
function load(group, table)

end