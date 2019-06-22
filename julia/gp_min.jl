using BayesianOptimization, GaussianProcesses, Distributions
using Distributed
using Serialization

function gp_minimize(f::Function, d::Int; verbose=true, file="gp_minimize",
                     iterations=400, acquisition="ei", noisebounds = [-4, 5])

    if acquisition isa String
        acquisition = Dict(
            "ei" => ExpectedImprovement(),
            "ucb" => UpperConfidenceBound()
        )[acquisition]
    end

    model = ElasticGPE(d,
      mean = MeanConst(0.),
      kernel = SEArd(zeros(d), 5.),
      logNoise = -2.,
      capacity = iterations
    )
    # set_priors!(model.mean, [Normal(1, 2)])

    iter = 0
    Xi = Vector{Float64}[]
    yi = Float64[]


    function g(x)
        iter += 1
        # print("($iter)  ")
        fx, elapsed = @timed f(x)
        verbose && println(
            "($iter)  ",
            round.(x; digits=3),
            " => ", round(fx; digits=4),
            "   ", round(elapsed; digits=1), " seconds",
            " with ", nprocs(), " processes"
        )
        push!(Xi, x)
        push!(yi, fx)
        open(file * "_xy", "w+") do file
            serialize(file, model)
        end
        fx
    end

    model_optimizer = MAPGPOptimizer(
        every = 20,
        noisebounds = noisebounds,       # bounds of the logNoise
        # kernbounds = [[-1, -1, 0], [4, 4, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
        maxeval = 100
    )

    opt = BOpt(
        g, model,
        acquisition,
        model_optimizer,
        zeros(d), ones(d),
        maxiterations = iterations,
        sense = Min,
        verbosity = Timings,
        initializer_iterations=cld(iterations, 4),
        repetitions=1,
    )

    res = boptimize!(opt)
    open(file * "_opt", "w+") do file
        serialize(file, opt)
    end
    opt
end



# f(x; noise=0.1) = sum((x .- 0.5).^2) + noise * randn()
