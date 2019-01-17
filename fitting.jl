using BlackBoxOptim
using Distributed

@everywhere include("model.jl")
include("human.jl")

const human_mean_fix = mean([length(t.fixations) for t in trials])
const human_mean_value = mean([t.value[t.choice] for t in trials])
const μ_emp, σ_emp = juxt(mean, std)(flatten(trials.value))

# %% ==================== Simulate experiment ====================
@everywhere function simulate(policy, value)
    cs = Int[]
    s = State(policy.m, value)
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c))
    (fixations=cs[1:end-1], choice=roll.choice, value=value)
end

function simulate_experiment(policy, μ, σ)
    pmap(trials.value) do v
        simulate(policy, (v .- μ) ./ σ)
    end
end

function summarize(pol, μ, σ)
    sim = simulate_experiment(pol, μ, σ);
    n_fix = map(sim) do s
        sum(diff(s.fixations) .!= 0) + 1
    end
    choice_val = map(trials, sim) do t, s
        t.value[s.choice]
    end
    (n_fix=mean(n_fix), choice_val=mean(choice_val))
end

# %% ==================== Optimize value prior ====================
m = MetaMDP(obs_sigma=5, sample_cost=0.001, switch_cost=5)
pol = MetaGreedy(m)

function optimize_prior(pol)
    function loss(x)
        μ, σ = x
        n_fix, choice_val = summarize(pol, μ, σ)
        sum(([n_fix, choice_val] - [human_mean_fix, human_mean_value]) .^ 2)
    end

    bounds = [(0., 20.), (.1, 5.)]
    res = bboptimize(loss; SearchRange=bounds, MaxTime=10.0, Method=:dxnes)
    μ, σ = best_candidate(res)
    (μ=μ, σ=σ, loss=loss([μ, σ]))
end



sim = simulate_experiment(pol, 8.8, 1.5)
mean_n_sample = mean([length(s.fixations) for s in sim])
mean_total_fix_time = mean(map(sum, trials.fix_times))
sample_time = mean_total_fix_time / mean_n_sample


max_n_fix = maximum(length(s.fixations) for s in sim)
X = zeros(3, max_n_fix)
for s in sim
    ranks = sortperm(sortperm(-s.value))
    for (j, f) in enumerate(s.fixations)
        r = ranks[f]
        X[r, j] += 1
    end
end
X ./= sum(X; dims=1)

f = plot(X');
hline!([1/3]);
f


num_fixation(cs) = sum(diff(cs) .!= 0) + 1
m = MetaMDP(obs_sigma=5, sample_cost=0.001, switch_cost=5)
x = simulate(m, trials[1].value)
