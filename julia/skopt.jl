using PyCall

@pyimport skopt
# Optimizer methods
ask(opt)::Vector{Float64} = opt[:ask]()
tell(opt, x::Vector{Float64}, fx::Float64) = opt[:tell](Tuple(x), fx)

py"""
import skopt

def expected_minimum(opt): 
    res = skopt.utils.create_result(opt.Xi, opt.yi, opt.space, opt.rng, models=opt.models)
    return skopt.utils.expected_minimum(res)
"""
expected_minimum = py"expected_minimum"