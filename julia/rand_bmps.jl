using Distributed
addprocs()
include("results.jl")
results = Results("halving/bmps/rand")
include("bmps_moments_fitting.jl")


let
    NTASK = isempty(ARGS) ? 4 : parse(Int, ARGS[1])

    xs = [rand(3) for i in 1:10000]

    best = Inf
    asyncmap(xs; ntasks=NTASK) do x
        # results = Results(NAME)
        fx = loss(x)
        if fx < best
            println("New best: ", x => fx)
            best = fx
        end
        # save(results, :xy, (x=x, y=fx))
    end
end