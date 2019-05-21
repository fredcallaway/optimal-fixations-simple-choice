using Distributed
addprocs();
@everywhere begin
    cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
    include("model.jl")
    include("job.jl")
    include("human.jl")
    include("simulations.jl")
    include("loss.jl")
end

using Glob
files = glob("runs/rando1000/jobs/*")
jobs = Job.(files)

@sync @distributed for job in jobs
    if exists(job, :simulations) && !exists(job, :losses)
        try
            sims = deserialize(job, :simulations)
            losses = map(sims) do sim
                sim.losses
            end
            serialize(job, :losses, losses)
        catch
            println("ERROR")
        end
    end
end