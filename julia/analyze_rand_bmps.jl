using Distributed
addprocs()
@everywhere include("model_base.jl")



# %% ====================  ====================
using Glob
all_policies = asyncmap(glob("results/foobar/*/policy")) do f
    open(deserialize, f)
end

# %% ====================  ====================
@everywhere function is_reasonable(policy)
    for i in 1:20
        steps = rollout(policy; max_steps=200).steps
        (steps == 1 || steps == 200) && return false
    end
    return true
end

reasonable = pmap(is_reasonable, all_policies)
policies = all_policies[reasonable];
@show mean(reasonable);
@show length(policies);

all_policies

# %% ==================== Simulation ====================

function simulate_experiment(policy, n_repeat=10, sample_time=100)
    sim = @distributed vcat for v in repeat(trials.value, n_repeat)
        sim = simulate(policy, (v .- μ_emp) ./ σ_emp)
        fixs, fix_times = parse_fixations(sim.samples, sample_time)
        (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times,)
    end
    Table(sim)
end

# @time sim = simulate_experiment(policies[1])

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

sim_loss = make_loss([choice_value, n_fix, fix_len])

# function loss(policy::Policy)
#     √(_sim_loss(simulate_experiment(policy)))
# end

# %% ====================  ====================
function asyncmap_bar(f, xs; kws...)
    print('.' ^ length(xs), "|\r")
    res = asyncmap(xs; kws...) do x
        fx = f(x)
        print('X')
        fx
    end
    println()
    res
end

# %% ====================  ====================


losses = asyncmap(policies; ntasks=10) do policy
    print('.')
    sim = simulate_experiment(policy, 1)
    sim_loss(sim)
end

# %% ====================  ====================

# %% ====================  ====================
minimum(losses)
best = policies[argmin(losses)]
open("tmp/best_bmps", "w") do f
    serialize(f, best)
end

sim = simulate_experiment(best, 10);

open("tmp/best_bmps_sim", "w") do f
    serialize(f, sim)
end

mean_reward(best, 10000, true)
@everywhere include("dc.jl")
mean_reward(Blinkered(best.m), 10000, false)






