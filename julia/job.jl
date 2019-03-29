# must be included after model.jl

using Parameters
using JSON
using Dates: now
using Serialization

@with_kw struct Job
    n_arm::Int = 2
    obs_sigma::Float64 = 1
    sample_cost::Float64 = 0.001
    switch_cost::Float64 = 1

    n_iter::Int = 200
    n_roll::Int = 1000
    seed::Int = 0
    group::String = "dummy"
end
function Job(d::Dict{String,Any})
    kws = Dict(Symbol(k)=>v for (k, v) in d)
    Job(;kws...)
end
function Job(file::AbstractString)
    Job(JSON.parsefile(file))
end
MetaMDP(job::Job) = MetaMDP(
    job.n_arm,
    job.obs_sigma,
    job.sample_cost,
    job.switch_cost,
)
Base.string(job::Job) = join((getfield(job, f) for f in fieldnames(Job)), "-")
result_file(job::Job, name) = "runs/$(job.group)/results/$name-$(string(job))"

function save(job::Job, name::Symbol, value)
    d = Dict(
        :job => job,
        :time => now(),
        :value => value
    )
    file = result_file(job, name) * ".json"
    open(file, "w") do f
        write(f, JSON.json(d))
    end
    serialize(job, name, value)
    println("Wrote $file")
    return file
end
load(job::Job, name::Symbol) = JSON.parsefile(result_file(job, name) * ".json")["value"]

function Serialization.serialize(job::Job, name::Symbol, value)
    file = result_file(job, name) * ".jls"
    open(file, "w") do f
        serialize(f, value)
    end
    println("Wrote $file")
end

function Serialization.deserialize(job::Job, name::Symbol)
    file = result_file(job, name) * ".jls"
    open(deserialize, file)
end

if length(ARGS) == 2
    job_group, job_id = ARGS
    const JOB = Job("runs/$job_group/jobs/$job_id.json")
    println(JOB)
else
    const JOB = nothing
end
