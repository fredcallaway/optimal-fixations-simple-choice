using Distributed
@everywhere begin
    include("model_base.jl")
    include("bmps.jl")
end
include("optimize_bmps.jl")
include("results.jl")
include("box.jl")
include("gp_min.jl")

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
    :σ_obs => (1, 5),
    :sample_cost => (1e-3, 1e-2, :log),
    :switch_cost => (1e-3, 5e-2, :log),
    :µ => μ_emp,
    :σ => σ_emp,
    :sample_time => 100
    # (0, 2 * μ_emp),
    # :σ => (σ_emp / 4, 4 * σ_emp),
)

if !@isdefined(results)
    results = Results("bmps_moments")
end
save(results, :space, space)

# %% ==================== Simulation ====================

function simulate_experiment(policy::Policy, n_repeat=100, sample_time=100)
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

descriptors = [choice_value, n_fix, total_fix_time]
loss_functions = make_loss.(descriptors)
multi_loss(sim) = [loss(sim) for loss in loss_functions]
sim_loss = make_loss(descriptors)

@info("Target descriptor values",
    choice_value = juxt(mean, std)(choice_value(trials)),
    n_fix = juxt(mean, std)(n_fix(trials)),
    total_fix_time = juxt(mean, std)(total_fix_time(trials)),
)

RECORD = (x=Vector{Float64}[], y=Float64[], policies=BMPSPolicy[])

function loss(prm::Params; verbose=false)
    m = MetaMDP(prm)
    policy, reward = optimize_bmps(m)
    push!(RECORD.policies, policy)
    sim = simulate_experiment(policy, 100)
    y = √(sim_loss(sim))
    @info("Loss",
        total=y,
        losses=multi_loss(sim),
        m,
        θ=round.(collect(policy.θ); digits=3),
        reward,
        mean(choice_value(sim)),
        mean(n_fix(sim)),
        mean(total_fix_time(sim)),
    )
    min(10., y)
end

function loss(x::Vector{Float64}; kws...)
    @debug "Compute loss" x
    y = loss(Params(space(x)); kws...)
    push!(RECORD.x, x)
    push!(RECORD.y, y)
    save(results, :record, RECORD; verbose=false)
    y
end

