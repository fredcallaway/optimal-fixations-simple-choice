using PyCall
using LatinHypercubeSampling: LHCoptim
using Serialization
using Distributed
skopt = pyimport("skopt")
# Optimizer methods
ask(opt)::Vector{Float64} = opt.ask()
tell(opt, x::Vector{Float64}, fx::Float64) = opt.tell(Tuple(x), fx)

py"""
import skopt

import warnings
warnings.filterwarnings("ignore")

def expected_minimum(opt):
    res = skopt.utils.create_result(opt.Xi, opt.yi, opt.space, opt.rng, models=opt.models)
    return skopt.utils.expected_minimum(res)
"""
expected_minimum = py"expected_minimum"

# using Memoize
# @memoize function get_latin_points(n_latin, dim)
#     LHCoptim(n_latin, length(bounds), 1000)[1] ./ n_latin
# end

function gp_minimize(f, dim, n_latin, n_bo; file="opt_xy")
    bounds = [(0., 1.) for i in 1:dim]
    opt = skopt.Optimizer(bounds, random_state=0, n_initial_points=n_latin)

    latin_points = LHCoptim(n_latin, length(bounds), 1000)[1] ./ n_latin

    iter = 1
    Xi = Vector{Float64}[]
    yi = Float64[]

    function g(x)
        # print("($iter)  ")
        fx, elapsed = @timed f(x)
        println("($iter)  ",
                round.(x; digits=3),
                " => ", round(fx; digits=4),
                "   ", round(elapsed; digits=1), " seconds",
                " with ", nprocs(), " processes"
                )
        iter += 1
        push!(Xi, x)
        push!(yi, fx)
        open(file, "w+") do file
            serialize(file, (Xi=Xi, yi=yi))
        end
        fx
    end

    # print("LHC: ")


    # for i in 1:n_latin
    asyncmap(1:n_latin; ntasks=3) do i
        x = latin_points[i, :]
        # print(".")
        tell(opt, x, g(x))
    end
    # print("\nBO: ")
    for i in 1:n_bo
        # print(".")
        x = ask(opt)
        tell(opt, x, g(x))
    end
    # print("\n")
    return opt
end