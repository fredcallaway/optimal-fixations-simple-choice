include("model.jl")
include("job.jl")


const N_SIM = 10
const human_mean_fixation_time = mean(sum.(trials.fix_times))
const μ_emp, σ_emp = juxt(mean, std)(flatten(trials.value))

# const human_mean_fix = mean([length(t.fixations) for t in trials])
# const human_mean_value = mean([t.value[t.choice] for t in trials])

# %% ==================== Simulate experiment ====================
function simulate(policy, value)
    cs = Int[]
    s = State(policy.m, value)
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c); max_steps=1000)
    (samples=cs[1:end-1], choice=roll.choice, value=value)
end


function parse_fixations(samples, time_per_sample)
    fixations = Int[]
    fix_times = Float64[]
    prev = nothing
    for x in samples
        if x != prev
            prev = x
            push!(fixations, x)
            push!(fix_times, 0.)
        end
        fix_times[end] += time_per_sample
    end
    fixations, fix_times
end

function simulate_experiment(policy, (μ, σ), n_repeat=N_SIM)
    samples, choice, value = map(1:n_repeat) do i
        map(trials.value) do v
            sim = simulate(policy, (v .- μ) ./ σ)
            sim.value[:] = v  # we want the un-normalized values
            sim
        end
    end |> flatten |> invert
    time_per_sample = human_mean_fixation_time / mean(length.(samples))
    fixs, fix_times = parse_fixations.(samples, time_per_sample) |> invert
    Table((choice=choice, value=value, fixations=fixs, fix_times=fix_times))
end

# %% ==================== Load policy for job ====================

function x2theta(x)
    cost_weight = x[1]
    voi_weights = diff([0; sort(collect(x[2:end])); 1])
    [cost_weight; voi_weights]
end

function optimized_policy(job)
    m = MetaMDP(job)
    try
        Policy(m, deserialize(job, :optim).θ1)
    catch
        missing
    end
end
