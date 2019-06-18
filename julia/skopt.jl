using PyCall
using LatinHypercubeSampling: LHCoptim
using Serialization
using Distributed
using SplitApplyCombine: splitdims

skopt = pyimport("skopt")
# Optimizer methods
ask(opt)::Vector{Float64} = opt.ask()
ask(opt, n::Int)::Vector{Vector{Float64}} = splitdims(opt.ask(n_points=n), 1)
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

using Memoize
@memoize function get_latin_points(n_latin, dim)
    LHCoptim(n_latin, dim, 1000)[1] ./ n_latin
end

const N_TASKS_LATIN = 3
const N_TASKS_BO = 3

function gp_minimize(f, dim, n_latin, n_bo; file="opt_xy")
    bounds = [(0., 1.) for i in 1:dim]
    opt = skopt.Optimizer(bounds, random_state=0, n_initial_points=n_latin)

    latin_points = get_latin_points(n_latin, dim)

    iter = 1
    # Xi = Vector{Float64}[]
    # yi = Float64[]

    function g(x)
        # print("($iter)  ")
        fx, elapsed = @timed f(x)
        println("($(length(opt.Xi)+1))  ",
                round.(x; digits=3),
                " => ", round(fx; digits=4),
                "   ", round(elapsed; digits=1), " seconds",
                " with ", nprocs(), " processes"
                )
        iter += 1
        # push!(Xi, x)
        # push!(yi, fx)
        # open(file, "w+") do file
        #     serialize(file, (Xi=Xi, yi=yi))
        # end
        fx
    end

    asyncmap(1:n_latin; ntasks=N_TASKS_LATIN) do i
        x = latin_points[i, :]
        tell(opt, x, g(x))
    end

    questions = Channel(N_TASKS_BO)
    answers = Channel(N_TASKS_BO)
    waiting = Set()

    for x in ask(opt, N_TASKS_BO)
        put!(questions, x)
    end

    function next_question()
        if isempty(waiting)
            return ask(opt)
        end

        q = ask(opt)
        if all(sum(abs.(q .- x)) > .01 * dim for x in waiting)
            # println("Different waiting")
            return q
        end

        # avoid asking duplicate questions by taking into account
        # previous questions currently awaiting answers
        fake_opt = opt.copy()
        for (i, x) in enumerate(collect(waiting))
            y = opt.models[end].predict([x])[1]
            fake_opt.tell(Tuple(x), y, fit=(i==length(waiting)))
        end
        return ask(fake_opt)
    end

    function master()
        while length(opt.Xi) < n_latin + n_bo
            x, fx = take!(answers)
            pop!(waiting, x)
            tell(opt, x, fx)
            if length(opt.Xi) + length(waiting) < n_latin + n_bo
                # TODO does this yield?
                put!(questions, next_question())
            end
        end
        close(questions)
        # close(answers)
    end

    function worker()
        while isopen(questions)
            x = take!(questions)
            push!(waiting, x)
            put!(answers, (x, g(x)))
            yield()
        end
    end

    for i in 1:N_TASKS_BO
        @async worker()
    end
    master()


    open(file, "w+") do file
        serialize(file, (Xi=map(collect, opt.Xi), yi=opt.yi))
    end
    println("gp_minimize complete")
    return opt
end