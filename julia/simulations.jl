using Distributed

function parse_fixations(samples; sample_time=100)
    fixations = Int[]
    fix_times = Float64[]
    prev = nothing
    for x in samples
        if x != prev
            prev = x
            push!(fixations, x)
            push!(fix_times, 0.)
        end
        fix_times[end] += sample_time
    end
    fixations, fix_times
end

function simulate(policy::Policy, prior::Tuple, value::Vector; max_steps=200)
    μ, σ = prior
    s = State(policy.m, (value .- μ) ./ σ)
    cs = Int[]
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c); max_steps=max_steps)
    fixs, fix_times = parse_fixations(cs[1:end-1])
    (choice=roll.choice, value=value, fixations=fixs, fix_times=fix_times)
end

function simulate_trials(policy::Policy, prior::Tuple, trials::Table)
    map(trials.value) do v
        simulate(policy, prior, v)
    end |> Table
end

# function simulate_experiment(policy::Policy, trials; μ, σ, n_repeat=1, sample_time=100)
#     @assert policy.m.n_arm == length(trials.value[1])  # policy must be trained on correct number of items
#     sim = @distributed vcat for v in repeat(trials.value, n_repeat)
#         sim = simulate(policy, (v .- μ) ./ σ)
#         fixs, fix_times = parse_fixations(sim.samples, sample_time)
#         (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times,)
#     end
#     Table(sim)
# end
