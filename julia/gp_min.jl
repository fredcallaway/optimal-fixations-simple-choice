using BayesianOptimization, GaussianProcesses, Distributions
using Distributed
using Serialization

function gp_minimize(f::Function, d::Int; kernel=Mat32Ard(zeros(d), 5.), verbose=true, init_Xy=nothing, run=true,
                     iterations=400, repetitions=1, acquisition_restarts=50,
                     optimize_every=20, noisebounds = [-4, 5], kernbounds=nothing,
                     init_iters=cld(iterations, 4),
                     acquisition="ei", noise=-2, mean=0.)

    if acquisition isa String
        acquisition = Dict(
            "ei" => ExpectedImprovement(),
            "ucb" => UpperConfidenceBound()
        )[acquisition]
    end



    model = ElasticGPE(d,
      mean = MeanConst(mean),
      kernel = kernel,
      logNoise = float(noise),
      capacity = iterations + (init_Xy != nothing ? size(init_Xy[1],2) : 0)
    )

    if init_Xy != nothing
        X, y = init_Xy
        y = -y  # because we are minimizing
        append!(model, X, y)
        init_iters = 0
    end
    # set_priors!(model.mean, [Normal(1, 2)])

    iter = 0
    function g(x)
        iter += 1
        fx, elapsed = @timed f(x)
        verbose && println(
            "($iter)  ",
            round.(x; digits=3),
            " => ", round(fx; digits=4),
            "   ", round(elapsed; digits=1), " seconds",
            " with ", nprocs(), " processes"
        )
        fx
    end

    if optimize_every <= 0
        model_optimizer = NoModelOptimizer()
    else
        model_optimizer = MAPGPOptimizer(
            every = optimize_every,
            noisebounds = noisebounds,       # bounds of the logNoise
            kernbounds = kernbounds,
            maxeval = 200
        )
    end

    opt = BOpt(
        g, model,
        acquisition,
        model_optimizer,
        zeros(d), ones(d),
        maxiterations = iterations,
        sense = Min,
        verbosity = verbose ? Timings : Silent,
        initializer_iterations=init_iters,
        repetitions=repetitions,
        acquisitionoptions=(restarts=acquisition_restarts,)
    )
    if run
        boptimize!(opt)
        find_model_max!(opt)
    end
    return opt
end

function find_model_max!(opt::BOpt; restarts=1000)
    opt.model_optimum, opt.model_optimizer = BayesianOptimization.acquire_model_max(opt;
        options=(method = :LD_LBFGS, restarts = restarts, maxeval = 2000)
    )
end

# f(x; noise=0.1) = sum((x .- 0.5).^2) + noise * randn()
