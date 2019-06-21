using BayesianOptimization, GaussianProcesses, Distributions
using Distributed
using Serialization

function gp_minimize(f, d; verbose=true, file="gp_opt", iterations=400)

    opt = nothing

    function g(x)
        # print("($iter)  ")
        fx, elapsed = @timed f(x)
        verbose && println(
            "($(opt.iterations.i))  ",
            round.(x; digits=3),
            " => ", round(fx; digits=4),
            "   ", round(elapsed; digits=1), " seconds",
            " with ", nprocs(), " processes"
        )
        if opt.iterations.i % 10 == 0
            open(file, "w+") do file
                serialize(file, opt)
            end
        end
        fx
    end

    model = ElasticGPE(d,
      mean = MeanConst(0.),
      kernel = SEArd(zeros(d), 5.),
      logNoise = -2.,
      capacity = iterations
    )

    model_optimizer = MAPGPOptimizer(
        every = 20,
        noisebounds = [-4, 5],       # bounds of the logNoise
        # kernbounds = [[-1, -1, 0], [4, 4, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
        maxeval = 100
    )
    acquisition = UpperConfidenceBound()

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
    open(file, "w+") do file
        serialize(file, opt)
    end
    opt
end

