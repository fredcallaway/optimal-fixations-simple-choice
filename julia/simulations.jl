using Distributed
# addprocs(40)
# if JOB == nothing
# pop!(ARGS)
# cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")

@everywhere include("model.jl")
include("job.jl")
include("human.jl")

const N_SIM = 10

const human_mean_fix = mean([length(t.fixations) for t in trials])
const human_mean_value = mean([t.value[t.choice] for t in trials])
const μ_emp, σ_emp = juxt(mean, std)(flatten(trials.value))

PARALLEL_SIM = false


# %% ==================== Simulate experiment ====================
@everywhere function simulate(policy, value)
    cs = Int[]
    s = State(policy.m, value)
    roll = rollout(policy, state=s, callback=(b,c)->push!(cs, c); max_steps=1000)
    (fixations=cs[1:end-1], choice=roll.choice, value=value)
end

function simulate_experiment(policy; μ=0, σ=σ_emp, n_repeat=N_SIM)
    mapmany(1:n_repeat) do i
        map(trials.value) do v
            sim = simulate(policy, (v .- μ) ./ σ)
            sim.value[:] = v  # we want the un-normalized values
            sim
        end
    end |> Table
end

function x2theta(x)
    cost_weight = x[1]
    voi_weights = diff([0; sort(collect(x[2:end])); 1])
    [cost_weight; voi_weights]
end

function optimized_policy(job)
    try
        m = MetaMDP(job)
        optim = load(job, :optim)
        # Policy(m, x2theta(X[argmin(y)]))
        Policy(m, x2theta(optim["x1"]))
    catch
        missing
    end
end


# %% ==================== Summary statistics ====================

# function summarize(sim)
#     n_fix = map(sim) do s
#         sum(diff(s.fixations) .!= 0) + 1
#     end
#     choice_val = map(trials, sim) do t, s
#         t.value[s.choice]
#     end
#     (n_fix=mean(n_fix), choice_val=mean(choice_val))
# end

struct Binning{T}
    limits::Vector{T}
end

Binning(xs, n) = begin
    lims = quantile(xs, range(0, 1, length=n+1))
    lims[end] *= 1.00001
    Binning(lims)
end

(bins::Binning)(x) = begin
    idx = findfirst(x .< bins.limits)
    idx == nothing ? nothing : idx - 1
end
bin(xs, n) = Binning(xs, n).(xs)

function bin_means(x, y; n=5)
    bins = bin(x, n)
    group(x->x[1], x->x[2], zip(bins, y)) |>
      sort |> values |> collect .|> mean
end

# %% ==================== Loss functions ====================
function norm_sse(y, yhat)
    sum(((y .- yhat) ./ std(y)) .^ 2)
end

function rel_mse(y, yhat)
    mean(((y .- yhat) ./ y) .^ 2)
end

function mpe(y, yhat)
    @assert size(y) == size(yhat)
    mean(abs.(y .- yhat) ./ y)
end
choice_vals(sim) = [s.value[s.choice] for s in sim]

function rt_val_std_loss(sim)
    n_sim = Int(length(sim) / length(trials))
    n_sample = length.(sim.fixations)
    time_per_sample = mean(fix_time) / mean(n_sample)
    pred_fix_time = n_sample * time_per_sample
    mpe(bin_means(val_std, fix_time), bin_means(repeat(val_std, n_sim), pred_fix_time))
end

function choice_val_std_loss(sim)
    n_sim = Int(length(sim) / length(trials))
    mpe(bin_means(val_std, human_choice_value), bin_means(repeat(val_std, n_sim), choice_vals(sim)))
end

function choice_val_max_loss(sim)
    n_sim = Int(length(sim) / length(trials))
    mpe(bin_means(val_max, human_choice_value), bin_means(repeat(val_std, n_sim), choice_vals(sim)))
end

function mean_choice_val_loss(sim)
    mpe(human_mean_value, mean(choice_vals(sim)))
end

human_n_fix = [length(t.fixations) for t in trials]
human_mean_n_fix = mean(human_n_fix)
function mean_n_fix_loss(sim)
    pred = map(sim) do s
        sum(diff(s.fixations) .!= 0) + 1
    end |> mean
    mpe(human_mean_n_fix, pred)
end

function hist_n_fix_loss(sim)
    max = floor(Int, quantile(human_n_fix, 0.95))
    pred = map(sim) do s
        sum(diff(s.fixations) .!= 0) + 1
    end
    mpe(counts(human_n_fix, 1:max) ./ length(human_n_fix), counts(pred, 1:max) ./ length(pred))
end

# fix_rank_target = hcat([bin_means(human_fix_ranks.time, human_fix_ranks.focus .== i; n=10) for i in (1,3)]...)
#
# function get_fix_ranks(sim)
#     n_sample = length.(sim.fixations)
#     time_per_sample = mean(fix_time) / mean(n_sample)
#     D = mapmany(sim) do s
#         ranks = sortperm(sortperm(-s.value))
#         enumerate(ranks[s.fixations])
#     end |> invert
#     T = Table(time=D[1] * time_per_sample, focus=D[2])
# end
#
# function fix_rank_loss(sim)
#     D = mapmany(sim) do s
#         ranks = sortperm(sortperm(-s.value))
#         enumerate(ranks[s.fixations])
#     end |> invert
#     pred = Table(step=D[1], focus=D[2])
#     pred = hcat([bin_means(pred.step, pred.focus .== i; n=10) for i in (1,3)]...)
#     if size(pred) != size(fix_rank_target)
#         return Inf
#     end
#     mpe(fix_rank_target, pred)
# end

function loss(sim)
    if ismissing(sim)
        return Inf
    end
    rt_val_std_loss(sim) +
    choice_val_max_loss(sim) +
    hist_n_fix_loss(sim)
    # fix_rank_loss(sim)
    # mean_choice_val_loss(sim)
end

# %% ==================== Optimize prior ====================
# using BlackBoxOptim

function make_prior_loss(pol)
    x -> begin
        μ, σ = x
        loss(simulate_experiment(pol; μ=μ, σ=σ))
    end
end

function optimize_prior(pol; max_func_evals=500)
    bounds = [(0., μ_emp), (.1, 5.)]
    my_loss = make_prior_loss(pol)
    options = []
    res = bboptimize(my_loss;
      SearchRange=bounds, Method=:dxnes, MaxFuncEvals=max_func_evals, PopulationSize=20,
      TraceInterval=10)
    μ, σ = best_candidate(res)
    (μ=μ, σ=σ)
end
