using Distributed
@everywhere begin
    include("model_base.jl")
    include("bmps.jl")
end
include("optimize_bmps.jl")
include("results.jl")
include("box.jl")

# %% ==================== Parameters ====================

@with_kw mutable struct Params
    α::Float64
    σ_obs::Float64
    sample_cost::Float64
    switch_cost::Float64
    µ::Float64
    σ::Float64
    sample_time::Float64
end
Params(d::AbstractDict) = Params(;d...)

MetaMDP(prm::Params) = MetaMDP(
    3,
    prm.σ_obs,
    prm.sample_cost,
    prm.switch_cost,
)

space = Box(
    :α => NaN,
    :σ_obs => (1, 20),
    :sample_cost => (1e-3, 1e-2, :log),
    :switch_cost => (1, 60),
    :µ => μ_emp,
    :σ => σ_emp,
    :sample_time => 100
    # (0, 2 * μ_emp),
    # :σ => (σ_emp / 4, 4 * σ_emp),
)

results = Results("moments/$(n_free(space))/bmps")
save(results, :space, space)

# %% ==================== Simulation ====================

function simulate_experiment(policy, n_repeat=10, sample_time=100)
    sim = @distributed vcat for v in repeat(trials.value, n_repeat)
        sim = simulate(policy, (v .- μ_emp) ./ σ_emp)
        fixs, fix_times = parse_fixations(sim.samples, sample_time)
        (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times,)
    end
    Table(sim)
end

# %% ==================== Loss ====================
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

total_fix_time(t)= sum(t.fix_times)
total_fix_time(tt::Table)= total_fix_time.(tt)

fix_len(tt::Table) = flatten(tt.fix_times)

function make_loss(descriptors::Vector{Function})
    losses = make_loss.(descriptors)
    (tt) -> sum(loss(tt) for loss in losses)
end

const sim_loss = make_loss([choice_value, n_fix, total_fix_time])

using Memoize
@memoize function bmps_policy(m::MetaMDP)
    policy, opt = optimize_bmps(m)
    return policy
end

function loss(prm::Params; no_memo=false, verbose=false)
    m = MetaMDP(prm)
    policy = no_memo ? optimize_bmps(m; verbose=verbose)[1] : bmps_policy(m)
    sim = simulate_experiment(policy, 10)
    min(10., √(sim_loss(sim)))
end

RECORD = (x=Vector{Float64}[], y=Float64[])

function loss(x::Vector{Float64}; kws...)
    y = loss(Params(space(x)); kws...)
    push!(RECORD.x, x)
    push!(RECORD.y, y)
    save(results, :record, RECORD; verbose=false)
    y
end

