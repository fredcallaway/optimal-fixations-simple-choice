# must be included after model.jl

using Parameters
using JSON

@with_kw struct Job
    n_arm::Int = 2
    obs_sigma::Float64 = 1
    sample_cost::Float64 = 0.001
    switch_cost::Float64 = 1

    n_iter::Int = 100
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
result_file(job::Job, name) = "runs/$(job.group)/results/$name-$(string(job)).json"

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
load(job::Job, name::Symbol) = JSON.parsefile(result_file(job, name))["value"]

if !isempty(ARGS)
    job_group, job_id = ARGS
    const JOB = Job("runs/$job_group/jobs/$job_id.json")
    println(JOB)
else
    const JOB = nothing
end
