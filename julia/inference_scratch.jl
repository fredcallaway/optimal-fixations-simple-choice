# %% ==================== Plot effect of parameters ====================
using Plots
function plot_feature(f, pol)
    x, y = f(simulate_experiment(pol, (μ_emp, σ_emp), 5))
    bins = make_bins(5, x)
    plot!(mids(bins), mean.(bin_by(bins, x, y)))
end

# xs = 10 .^ range(-5, -2, length=100)
# xs = range(.001, .1, length=100)
xs = range(1, 10, length=100)
n = 1000

ys = map(xs) do x
    m = MetaMDP(obs_sigma=x, switch_cost=4, sample_cost=1e-3)
    policy = MetaGreedy(m)
    y = @distributed (+) for i in 1:n
        sim = simulate(policy, randn(3))
        # length(sim.samples)
        sim.value[sim.choice] - maximum(sim.value)

    end
    y / n
end

map(trials) do t
    (maximum(t.value) - t.value[t.choice]) / σ_emp
end |> mean


# %% ==================== Compare Remi's method to SIR ====================
x1 = pmap(1:1000) do i
    old_logp(true_policy, 0.1, d.value, d.samples, d.choice; n_particle=Int(1e5))
end
x2 = pmap(1:1000) do i
    logp(true_policy, 0.1, d.value, d.samples, d.choice)
end
std(x1) / std(x2)
mean(x1) / mean(x2)

# %% ==================== Describe ====================

describe(f, label, agg=mean) = println("$label: ", agg(map(f, sim_data)))
display("")
describe("value loss") do d
    maximum(d.value) - d.value[d.choice]
end
describe("# samples") do d
    length(d.samples)
end
# describe("first fixation", x->quantile(x, 0.75)) do d
#     parse_fixations(d.samples, 100)[2][1]
# end


# %% ==================== Who knows what? ====================
# include("features.jl")
function plot_feature(f, pol)
    x, y = f(simulate_experiment(pol, (μ_emp, σ_emp), 5))
    bins = make_bins(5, x)
    plot!(mids(bins), mean.(bin_by(bins, x, y)))
end
plot()
f = value_choice

plot_feature(f, candidates[idx[1]])
plot_feature(f, candidates[idx[2]])
plot_feature(f, candidates[idx[3]])
# plot_feature(value_bias, candidates[idx[2]])

# %% ====================  ====================
display("")
function foo(pol)
    println("")
    println(pol.m)
    for _ in 1:20
        sim = simulate(pol, [0., 0.5, 1.])
        println(sim.choice, "  ", sim.samples)
    end
end
foo(MetaGreedy(true_mdp))
# foo(candidates[idx[1]])

# %% ====================  ====================
params = Base.product(2:6, 0.003:0.001:0.007, 2:6) |> collect
logp_grid = pmap(params) do x
    ε = 0.1
    m = MetaMDP(3, x...)
    mapreduce(+, sim_data) do d
        logp(m, ε, d)
    end
end


# %% ====================  ====================

params[3, :, :]
X = map(logp_grid) do x
    isfinite(x) ? x : NaN
end
pyplot()
heatmap(X[3, :, :], clim=(-1000, -750))

X = reshape(maximum(logp_grid, dims=1), (5, 5))
heatmap(X, clim=(-800, -750))

# maximum(logp_grid, dims=(1,3)) |> flatten |> plot

# %% ====================  ====================
using Glob
files = glob("runs/rando/jobs/*")
jobs = Job.(files)

@everywhere function logp(policy, ε, t::Trial)
    samples = discretize_fixations(t, sample_time=100)
    value = (t.value .- μ_emp) ./ σ_emp
    logp(policy, ε, value, samples, t.choice)
end

policies = map(optimized_policy, jobs) |> skipmissing |> collect
@everywhere dd = group(x->x.subject, trials)[18]
@time res = pmap(policies) do policy
    map(dd) do t
        logp(policy, 0.3, t)
    end
end

# %% ====================  ====================
@everywhere t = dd[1]
@time x = pmap(policies) do policy
    logp(policy, 0.3, t)
end

best = argmax(x)
pol = policies[best]

discretize_fixations(t, sample_time=100)
# %% ==================== Q regression? ====================


function foobar()
    N = 480
    v = [0., 0., 1.]
    x = @distributed (+) for i in 1:N
        rollout(policy; state=State(policy.m, v)).reward
    end
    x / N
end
foobar()




# %% ====================  ====================
function rand_logp(t::Trial)
    samples = discretize_fixations(t, sample_time=100)
    log(1/4) * (length(samples) + 1) + log(1/3)
end

function stay_logp(t::Trial, ε)
    samples = discretize_fixations(t, sample_time=100)
    switch = diff(samples) .!= 0
    p_first = log(1/3)
    p_stay = log(1-ε) * sum(.!switch)
    p_switch = log(ε * 1/4) * sum(switch)
    p_end = log(ε * 1/4)
    return p_first + p_stay + p_switch + p_end
end
stay_logp(t, 0.3)

struct StayPolicy
    m::MetaMDP
end
(π::StayPolicy)(b::Belief) = b.focused
logp(StayPolicy(MetaMDP(obs_sigma=1e10)), 0.3, t)

# %% ====================  ====================
best = argmax(map(sum, res))
map(sum, res)
simulate(policies[best], dd[4].value)











# %% ====================  ====================
steps, choices = repeatedly(10) do
    roll = rollout(MetaGreedy(m), state=s)
    (roll.steps, roll.choice)
end |> invert

proportions(choices, 3) |> Tuple

pf = ParticleFilter(m, 0.1, t)
d = parse_computations(t; sample_time=100)
history = []
ps = run!(pf, d.cs, callback=(particles->push!(history, deepcopy(particles))))
# %% ====================  ====================
using Plots

function plot_particles(i)
    c = d.cs[i]
    h = history[i]
    mu = combinedims([p.x.mu for p in h])
    w = [p.w for p in h]
    w ./= maximum(w)
    plot(mu, label="", color=:black, α=0.1w', ylim=(-1, 1), title=string(c))
end
# plots = [plot_particles(i) for i in 1:length(history)]
# plot(plots...)

# %% ====================  ====================



cs = parse_computations(t; sample_time=200)
x = counts(cs)[2:end]
x / sum(x)

groupreduce(x->x[1], x->x[2], +, zip(t.fixations, t.fix_times))