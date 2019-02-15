using BlackBoxOptim
using Distributed
using Printf

@everywhere include("model.jl")

seed = 0; n_iter = 100; n_roll = 000
Random.seed!(seed)
m = MetaMDP(n_arm=3, obs_sigma=7, sample_cost=0.001, switch_cost=8)
verbose = false
bounds = [ (0., max_cost(m)), (0., 1.), (0., 1.) ]

function sum_reward(policy; n_roll=1000, seed=0)
    @distributed (+) for i in 1:n_roll
        Random.seed!(seed + i)
        rollout(policy, max_steps=100).reward
    end
end

function avg_reward(policy; n_roll=1000, seed=0)
    sum_reward(policy, n_roll=n_roll, seed=seed) / n_roll
end

function loss(x; seed=0)
    policy = Policy(m, x2theta(x))
    reward, secs = @timed @distributed (+) for i in 1:n_roll
        Random.seed!(seed + i)
        rollout(m, policy, max_steps=100).reward
    end
    reward /= n_roll
    if verbose
        print("θ = ", round.(x2theta(x); digits=2), "   ")
        @printf "reward = %.3f   seconds = %.3f\n" reward secs
        flush(stdout)
    end
    - reward
end

function noisy_loss(x)
    policy = Policy(m, x2theta(x))
    reward, secs = @timed @distributed (+) for i in 1:n_roll
        rollout(m, policy, max_steps=100).reward
    end
    reward /= n_roll
    if verbose
        print("θ = ", round.(x2theta(x); digits=2), "   ")
        @printf "reward = %.3f   seconds = %.3f\n" reward secs
        flush(stdout)
    end
    - reward
end

for i in 1:10
    x = ask(opt)
    @time y = noisy_loss(x)
    @time tell(opt, x, y)
end




avg_reward(Policy(m, x2theta(x)); n_roll=5000)

noisy_loss(x)
res = tell(opt, x, y)

@time noisy_loss(x)
@time tell(opt, x, noisy_loss(x))

x = rand(3) .* [b[2] for b in bounds]
loss(x)
dxnes_res = bboptimize(loss; SearchRange=bounds, MaxTime=10.0, Method=:dxnes)
loss(best_candidate(res2); seed=0)
loss(best_candidate(res2); seed=10000)



# %% ====================  ====================

function halving(m::MetaMDP)
    n = 500
    bounds = [ (0., max_cost(m)), (0., 1.), (0., 1.) ]
    pop = [Policy(m, x2theta(rand(3) .* [b[2] for b in bounds])) for i in 1:n]

    n_eval = zeros(Int, n)
    score = zeros(n)
    avg_score = zeros(n)
    history = []

    reduction = 2
    for t in 0:7
        r = 100 * reduction ^ t
        q = 1 - 1 / reduction ^ t
        active = avg_score .>= quantile(avg_score, q)
        for i in 1:n
            if active[i]
                score[i] += sum_reward(pop[i]; n_roll=r, seed=n_eval[i])
                n_eval[i] += r
            end
        end
        avg_score = score ./ n_eval
        push!(history, avg_score)
        println(maximum(avg_score), "  ", maximum(n_eval))
    end
    return pop, avg_score, n_eval, history
end
@time pop, avg_score, n_eval, history = halving(m)


final = findall(n_eval .== maximum(n_eval))
winner = pop[final[argmax(avg_score[final])]]

avg_reward(pol; seed=100000, n_roll=10000)
avg_reward(winner; seed=100000, n_roll=10000)


# %% ====================  ====================
include("job.jl")

using Glob

job = Job(n_arm=3, obs_sigma=7.0, sample_cost=0.001, switch_cost=8., n_iter=100, seed=0, group="no-cv")
X, y = values(load(j, :optim))

f = glob("runs/emin/results/optim*")[5]

function parse_result(f)
    d = JSON.parsefile(f)
    job = Job(d["job"])
    m = MetaMDP(job)
    v = d["value"]
    (n_arm=job.n_arm,
     obs_sigma=job.obs_sigma,
     sample_cost=job.sample_cost,
     switch_cost=job.switch_cost,
     seed=job.seed,
     y=v["y1"],
     cv=avg_reward(Policy(m, x2theta(v["x1"])); n_roll=10000))
end

results = parse_result.(glob("runs/emin/results/optim*"))
using SplitApplyCombine

diffs = valmap(group(r->(r.n_arm, r.obs_sigma, r.sample_cost, r.switch_cost), results)) do g
    abs(g[1].cv - g[2].cv)
end |> values

maximum(diffs)
mean(diffs)

ranks = sortperm(y)
top20 = [Policy(m, x2theta(x)) for x in X[ranks[1:20]]]
pol = top20[1]

n_fixation(s) = sum(diff(s.focused) .!= 0) + 1

sim = simulate_experiment(pol, k=0)
mean(n_fixation.(sim))
mean([t.value[t.choice] for t in sim])

avg_reward(top20[1]; seed=0, n_roll=5000)
avg_reward(pol; seed=1000, n_roll=10000)
bb_res = bboptimize(loss, SearchRange=bounds)

avg_reward(Policy(m, x2theta(best_candidate(bb_res))); seed=1000, n_roll=5000)
avg_reward(pol; seed=1000, n_roll=5000)
avg_reward(Policy(m, x2theta(res[3])); seed=1000, n_roll=5000)

x2theta(res[3])
top20[5].θ
x2theta(best_candidate(bb_res))

include("hyperband.jl")
sample_x() = rand(3) .* [b[2] for b in bounds]
function get_loss(x, n)
    -avg_reward(Policy(m, x2theta(x)); n_roll=n)
end
res = halving(sample_x, get_loss, 2^10; r=100, reduction=2)

loss(res[3])

# %% ====================  ====================

function rosenbrock2d(x)
  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function noisy_rosenbrock(x, n)
    y = rosenbrock2d(x)
    n * (y + randn() / √n)
end


sample_x() = rand(2) .* 5
res = halving(sample_x, getloss, 10, 100, 2)
history
rosenbrock2d(res[3])

length(history)
res = hyperband(sample_x, getloss, 1000, 3)



