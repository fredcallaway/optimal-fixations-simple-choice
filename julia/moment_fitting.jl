using Parameters
using Optim

include("elastic.jl")

# if get(ARGS, 1, "") == "master"
include("results.jl")
include("box.jl")
include("gp_min.jl")
nprocs() == 1 && addprocs(topology=:master_worker)
println(nprocs(), " processes")

@everywhere begin
    include("model.jl")
    include("blinkered.jl")
    include("human.jl")
    include("simulations.jl")
end

# %% ==================== Parameters ====================

@with_kw mutable struct Params
    α::Float64
    obs_sigma::Float64
    sample_cost::Float64
    switch_cost::Float64
    µ::Float64
    σ::Float64
    sample_time::Float64
end
Params(d::AbstractDict) = Params(;d...)

MetaMDP(prm::Params) = MetaMDP(
    3,
    prm.obs_sigma,
    prm.sample_cost,
    prm.switch_cost,
)
SoftBlinkered(prm::Params) = SoftBlinkered(MetaMDP(prm), prm.α)

space = Box(
    # :α => (10, 100, :log),
    :α => 1e10,
    :obs_sigma => (1, 10),
    # :sample_cost => (0.0004, 0.002, :log),
    :sample_cost => (1e-4, 1e-2, :log),
    :switch_cost => (1, 60),
    :µ => μ_emp,
    :σ => σ_emp,
    :sample_time => 100
    # (0, 2 * μ_emp),
    # :σ => (σ_emp / 4, 4 * σ_emp),
)

# %% ==================== Simulation ====================

function simulate_experiment(prm::Params, n_repeat=100)
    policy = SoftBlinkered(prm)
    @unpack μ, σ, sample_time = prm
    sim = @distributed vcat for v in repeat(trials.value, n_repeat)
        sim = simulate(policy, (v .- μ) ./ σ)
        fixs, fix_times = parse_fixations(sim.samples, sample_time)
        (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times,)
    end
    Table(sim)
end

# %% ====================  ====================
function make_loss(descriptor::Function)
    human_mean, human_std = juxt(mean, std)(descriptor(trials))
    (tt) -> begin
        dtt = descriptor(tt)
        model_mean = isempty(dtt) ? 0. : mean(dtt)
        ((model_mean - human_mean) / human_std)^2
    end
end
choice_value(t) = t.value[t.choice]
choice_value(tt::Table) = choice_value.(tt)

n_fix(t) = length(t.fixations)
n_fix(tt::Table) = n_fix.(tt)

fix_len(tt::Table) = flatten(tt.fix_times)

function make_loss(descriptors::Vector{Function})
    losses = make_loss.(descriptors)
    (tt) -> sum(loss(tt) for loss in losses)
end

const sim_loss = make_loss([choice_value, n_fix, fix_len])

function loss(x::Vector{Float64})
    prm = Params(space(x))
    √(sim_loss(simulate_experiment(prm)))
end

prior(prm::Params) = (prm.μ, prm.σ)

# %% ====================  ====================

prepare_result(prm::Params) = (
    policy = SoftBlinkered(prm),
    prior = prior(prm),
    sample_time = prm.sample_time
)

function random_search(N=1000)
    results = Results("moments/$(n_free(space))/rand")
    save(results, :space, space)

    xs = [rand(3) for i in 1:N]
    ys = asyncmap(loss, xs; ntasks=10)
    best_x = xs[argmin(ys)]

    prm = Params(space(best_x))
    best = prepare_result(prm)
    save(results, :xy, (x=xs, y=ys))
    save(results, :best, best)
    return best
end

function gp_min(N=400)
    results = Results("moments/$(n_free(space))/gp_min")
    save(results, :space, space)
    opt = gp_minimize(loss, n_free(space), noisebounds=[-4, -2], iterations=N)
    println("observed: ", round.(opt.observed_optimizer; digits=3),
            " => ", round(opt.observed_optimum; digits=5))
    println("model:    ", round.(opt.model_optimizer; digits=3),
            " => ", round(opt.model_optimum; digits=5))
    f_mod = @show loss(opt.model_optimizer)
    f_obs = @show loss(opt.observed_optimizer)
    best_x = f_obs < f_mod ? opt.observed_optimizer : opt.model_optimizer

    prm = Params(space(best_x))
    best = prepare_result(prm)
    save(results, :opt, opt)
    save(results, :model, opt.model)
    save(results, :xy, (x=opt.model.x, y=opt.model.y))
    save(results, :best, best)
    return best
end

# %% ====================  ====================
function particles()
    results = Results("moments/$(n_free(space))/particles")
    save(results, :space, space)

    method = ParticleSwarm(
        lower = zeros(n_free(space)),
        upper = ones(n_free(space)),
        n_particles = 50)

    Xi = Vector{Float64}[]
    yi = Float64[]
    iter = 0

    f(x) = begin
        iter += 1
        fx = loss(x)
        println(
            "($iter)  ",
            round.(x; digits=3),
            " => ", round(fx; digits=4)
        )
        push!(Xi, x)
        push!(yi, fx)
        if iter % 10 == 0
            save(results, :xy, (x=Xi, y=yi))
        end
        fx
    end

    res = optimize(f, zeros(3), method, Optim.Options(f_tol=0.001))
    prm = Params(space(res.minimizer))
    save(results, :best, prepare_result(prm))
    save(results, :res, res)
end

