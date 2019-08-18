include("results.jl")
results = Results("halving/bmps/rand")
include("bmps_moments_fitting.jl")

using Logging; global_logger(SimpleLogger(stdout, Logging.Debug))

let
    x = rand(3)
    @time loss(x)
end
