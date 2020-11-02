using Serialization

@everywhere include("compute_likelihood.jl")

jobs = eval(Meta.parse(ARGS[1]))
pmap(jobs) do job
    map(1:7) do prior_i
        try
            do_job(compute_likelihood, "likelihood/$prior_i", job, prior_i)
        catch e
            println("ERROR on $job ", e)
        end
    end
end