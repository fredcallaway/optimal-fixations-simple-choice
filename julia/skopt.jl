using PyCall
using LatinHypercubeSampling

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


function gp_minimize(f, dim, n_latin, n_bo)
    bounds = [(0., 1.) for i in 1:dim]
    opt = skopt.Optimizer(bounds, random_state=0, n_initial_points=n_latin)

    latin_points = LHCoptim(n_latin, length(bounds), 1000)[1] ./ n_latin

    print("LHC: ")
    for i in 1:n_latin
        x = latin_points[i, :]
        print(".")
        tell(opt, x, f(x))
    end
    print("\nBO: ")
    for i in 1:n_bo
        print(".")
        x = ask(opt)
        tell(opt, x, f(x))
    end
    print("\n")
    return opt
end