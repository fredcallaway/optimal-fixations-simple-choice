include("model.jl")
include("job.jl")
using HDF5
using Distributed

# Rollouts.
@everywhere function tracked_rollout(policy::Policy; max_steps=500, value=nothing)
    m = policy.m
    s = State(m)
    b = Belief(s)
    if value != nothing
        s.value[:] = value
    end
    reward = 0
    focused = Int[]
    for step in 1:max_steps
        a = (step == max_steps) ? TERM : policy(b)
        reward += step!(m, b, s, a)
        push!(focused, b.focused)
        if a == TERM
            return (value=s.value, belief=b.mu, focused=focused, choice=argmax(b.mu), reward=reward)
        end
    end
end

function simulate_experiment(job::Job)
    Random.seed!(0)
    m = MetaMDP(job)
    policy = Policy(m, Array{Float64}(load(job, :optim)["theta"]))

    V = h5read("data.h5", "values")
    @time rollouts = Dict(
        k => [tracked_rollout(m, policy; value=V[:, j] .+ k) for j in 1:size(V)[2]]
        for k in 0:1:1
    )

    save(job, :matched_rollouts, rollouts)
    flush(stdout)
end

simulate_experiment(JOB)