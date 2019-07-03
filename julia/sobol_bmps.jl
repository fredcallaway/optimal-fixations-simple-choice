using Distributed
using Sobol

addprocs()
include("bmps_moments_fitting.jl")

START = 1
END = 1000
N_WORKER = 5

mkpath("results/sobol_bmps/started")
mkpath("results/sobol_bmps/done")

jobs = Channel(N_WORKER)

function suggestor()
    seq = SobolSeq(3)
    if START > 1
        skip(seq, START-1)
    end
    for i in START:END
        put!(jobs, (i, next!(seq)))
    end
    close(jobs)
end

function worker()
    for (name, x) in jobs
        started = "results/sobol_bmps/started/$name"
        isfile(started) && continue
        touch(started)
        println(x)
        sleep(rand())
        open("results/sobol_bmps/done/$name", "w+") do f
            serialize(f, (x=x, y=loss(x)))
        end
    end
end

@sync begin
    @async suggestor()

    for i in 1:N_WORKER
        @async worker()
    end
end
