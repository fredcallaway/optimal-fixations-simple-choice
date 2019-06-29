using Plots
plot([1,2])
# include("inference.j")
# addprocs()
# @everywhere include("inference.jl")
# include("results.jl")
# include("box.jl")
# %% ====================  ====================
pwd()
include("results.jl")
results = get_results("investigate/100")[end]
heatmap(load(results, :prior_grid))
xlabel!("sigma")
yticks!(1:11, map(string, 0:0.2:2))
ylabel!("mu")

# %% ====================  ====================

opt = open(deserialize, "results/2019-06-01T11-36-33/opt_xy")
using GaussianProcesses
X = opt.Xi

using JSON
open("results/2019-06-01T11-36-33/xy.json", "w") do f
    write(f, json((X=opt.Xi, y=opt.yi)))
end
    
best = opt.Xi[argmin(opt.yi)]

diffs = map(opt.Xi) do x
    abs.(x .- best)
end

results/2019-06-01T11-36-33/
best

using SplitApplyCombine
using RollingFunctions
x = randn(1000)
plot(rollmean(x, 100))

params = ["α", "obs_sigma", "sample_cost", "switch_cost", "µ", "σ"]
plot(rollmean.(invert(diffs), 10), label=params)


# %% ====================  ====================
using Glob
res = glob("results/*/opt_xy")
ymin = map(res) do r
    minimum(open(deserialize, r).yi)
end
idx = sortperm(ymin)
for i in idx
    println(res[i], "  ", ymin[i])
end

# %% ====================  ====================
X = map(Iterators.product([1,2,3], [10, 20, 30])) do (x, y)
    x + y
end
heatmap(X)
xlabel!("y")
ylabel!("x")
# %% ====================  ====================
results = get_results("recovery/fit6")[end]
true_prm = load(results, :true_prm)
space = load(results, :space)

# %% ====================  ====================
const N_PARTICLE = 500

function plogp(prm, particles=N_PARTICLE)
    smap(eachindex(data)) do i
        logp(prm, data[i], particles)
    end |> sum
end

function loss(x, particles=N_PARTICLE)
    prm = Params(;space(x)...)
    min(MAX_LOSS, plogp(prm, particles) / RAND_LOGP)
    # min(MAX_LOSS, -plogp(prm, particles) / N_OBS)
end


plogp(true_prm)
