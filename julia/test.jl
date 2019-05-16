# cd("/usr/people/flc2/juke/choice-eye-tracking/julia")
# include("model.jl")
using Turing

pwd()
cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
include("/usr/people/flc2/juke/choice-eye-tracking/julia/model.jl")


include("model.jl")
open("model.jl", "r") do f
    println(readlines(f))
end
``
