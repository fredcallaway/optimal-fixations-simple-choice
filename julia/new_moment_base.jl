include("box.jl")
include("meta_mdp.jl")
include("bmps.jl")
include("optimize_bmps.jl")
include("ucb_bmps.jl")
include("human.jl")

const MAX_STEPS = 200  # 20 seconds
const SAMPLE_TIME = 100

space = Box(
    :sample_time => 100,
    :α => (50, 200),
    :σ_obs => (1, 5),
    :sample_cost => (.001, .01),
    :switch_cost => (.01, .05),
)

x2prm(x) = x |> space |> namedtuple

# %% ==================== Data ====================

function build_dataset(num; fold="odd")
    trials = load_dataset(num)
    train, test = train_test_split(trials, fold)
    μ_emp, σ_emp = empirical_prior(trials)
    (
        n_item = length(trials[1].value),
        train_trials = train,
        test_trials = test,
        μ_emp = μ_emp,
        σ_emp = σ_emp,
    )
end
Dataset = NamedTuple{(:n_item, :train_trials, :test_trials, :μ_emp, :σ_emp)}

# %% ==================== Simulation ====================

function simulate(policy, value; max_steps=1000)
    cs = Int[]
    s = State(policy.m, value)
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c); max_steps=max_steps)
    (samples=cs[1:end-1], choice=roll.choice, value=value)
end


function sim_one(policy, μ, σ, v)
    sim = simulate(policy, (v .- μ) ./ σ; max_steps=MAX_STEPS)
    fixs, fix_times = parse_fixations(sim.samples, SAMPLE_TIME)
    (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times)
end
Sim = NamedTuple{(:choice, :value, :fixations, :fix_times),Tuple{Int64,Array{Int64,1},Array{Int64,1},Array{Float64,1}}}


function simulate(policy::Policy, ds::Dataset, β_μ; n_repeat=10)
    μ = β_μ * ds.μ_emp
    σ = ds.σ_emp
    sims = Sim[]
    for v in ds.train_trials.value
        for i in 1:n_repeat
            push!(sims, sim_one(policy, μ, σ, v))
        end
    end
    sims
end

function simulate(policies::Vector{T} where T <: Policy, ds::Dataset, β_μ)
    mapmany(policies) do pol
        simulate(pol, ds, β_μ; n_repeat=1)
    end
end

# %% ==================== Loss ====================

function total_fix_times(t)
    x = zeros(length(t.value))
    for i in eachindex(t.fixations)
        fi = t.fixations[i]; ti = t.fix_times[i]
        x[fi] += ti
    end
    return x
end

function relative(x)
    a = length(x); b = a - 1
    mx = sum(x) ./ b
    @. a/b * x - mx
end

function make_loss(descriptor::Function, ds::Dataset)
    human_mean, human_std = juxt(mean, std)(descriptor.(ds.train_trials))
    (tt) -> begin
        dtt = descriptor.(tt)
        model_mean = isempty(dtt) ? 0. : mean(dtt)
        ((model_mean - human_mean) / human_std)^2
    end
end

function make_loss(descriptors::Vector{Function}, ds::Dataset)
    losses = make_loss.(descriptors, [ds])
    (tt) -> sum(loss(tt) for loss in losses)
end


choice_value(t) = t.value[t.choice]
n_fix(t) = length(t.fixations)
total_fix_time(t) = sum(t.fix_times)
chosen_relative_fix(t) = relative(total_fix_times(t))[t.choice]
