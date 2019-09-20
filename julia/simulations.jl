# include("model.jl")
# include("job.jl")


const N_SIM = 10
# const human_mean_fixation_time = mean(sum.(trials.fix_times))

# const human_mean_fix = mean([length(t.fixations) for t in trials])
# const human_mean_value = mean([t.value[t.choice] for t in trials])

# %% ==================== Simulate experiment ====================
function simulate(policy, value; max_steps=1000)
    cs = Int[]
    s = State(policy.m, value)
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c); max_steps=max_steps)
    (samples=cs[1:end-1], choice=roll.choice, value=value)
end


function parse_fixations(samples, sample_time)
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

# function simulate_experiment(policy, (μ, σ), n_repeat=N_SIM;
#                              parallel=false, sample_time=nothing)
#     mymap = parallel ? pmap : map
#     samples, choice, value = map(1:n_repeat) do i
#         mymap(trials.value) do v
#             sim = simulate(policy, (v .- μ) ./ σ)
#             sim.value[:] = v  # we want the un-normalized values
#             sim
#         end
#     end |> flatten |> invert
#     if sample_time == nothing
#         sample_time = human_mean_fixation_time / mean(length.(samples))
#     end
#     fixs, fix_times = parse_fixations.(samples, sample_time) |> invert
#     Table((choice=choice, value=value, fixations=fixs, fix_times=fix_times))
# end

function simulate_experiment(policy::Policy, μ=μ_emp, σ=σ_emp; n_repeat=100, sample_time=100)
    sim = @distributed vcat for v in repeat(trials.value, n_repeat)
        sim = simulate(policy, (v .- μ) ./ σ)
        fixs, fix_times = parse_fixations(sim.samples, sample_time)
        (choice=sim.choice, value=v, fixations=fixs, fix_times=fix_times,)
    end
    Table(sim)
end
