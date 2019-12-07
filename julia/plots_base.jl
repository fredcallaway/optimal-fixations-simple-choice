using Distributed
nprocs() == 1 && addprocs()
@everywhere begin
    cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
    include("model_base.jl")
    include("dc.jl")
    include("features.jl")
end
using Serialization
using StatsPlots
plot([1,2])

# %% ====================  ====================
pyplot(label="")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
using Printf

RED = colorant"#FF6167"

NO_RIBBON = false
FAST = false
const CI = 0.95
const N_BOOT = 1000
using Bootstrap
function ci_err(y)
    NO_RIBBON & return (0., 0.)
    FAST && return (sem(y) * 2, sem(y) * 2)
    isempty(y) && return (NaN, NaN)
    bs = bootstrap(mean, y, BasicSampling(N_BOOT))
    c = confint(bs, BCaConfInt(CI))[1]
    abs.(c[2:3] .- c[1])
end

function plot_human!(bins, x, y, type=:line; kws...)
    vals = bin_by(bins, x, y)
    err = ci_err.(vals)
    if type == :line
        plot!(mids(bins), mean.(vals), yerr=err,
              grid=:none,
              line=(2,),
              color=:black,
              label="";
              kws...)
    elseif type == :discrete
        Plots.bar!(mids(bins), mean.(vals),
            yerr=err,
              grid=:none,
              fill=:white,
              line=(2,),
              color=:black,
              label="";
              kws...)
  else
      error("Bad plot type : $type")
  end
end


function plot_human!(feature::Function, bins=nothing, type=:line; kws...)
    hx, hy = feature(trials)
    bins = make_bins(bins, hx)
    plot_human!(bins, hx, hy, type; kws...)
end

function plot_model!(bins, x, y, type=:line; kws...)
    vals = bin_by(bins, x, y)
    err = invert(ci_err.(vals))
    plot_model!(mids(bins), mean.(vals), err, type; kws...)
end

function plot_model!(x::Vector{Float64}, y, err, type; color=RED, kws...)
    if type == :line
        plot!(x, y,
              ribbon=err,
              fillalpha=0.1,
              color=color,
              linewidth=1,
              label="";
              kws...)
    elseif type == :discrete
        plot!(x, y,
              yerr=err,
              grid=:none,
              color=color,
              linewidth=1,
              marker=(7, :diamond, color, stroke(0)),
              label="";
              kws...)
    else
        error("Bad plot type : $type")
    end
end


function cross!(x, y)
    vline!([x], line=(:grey, 0.7), label="")
    hline!([y], line=(:grey, 0.7), label="")
end

function plot_comparison(feature, trials, sim, bins=nothing, type=:line; kws...)
    plot()
    hx, hy = feature(trials; kws...)
    mx, my = feature(sim; kws...)
    bins = make_bins(bins, hx)
    plot_human!(bins, hx, hy, type)
    plot_model!(bins, mx, my, type)
    # title!(@sprintf "Loss = %.3f" make_loss(feature, bins)(sim))
end

function fig(f, name)
    _fig = f()
    savefig("figs/$run_name/$name.pdf")
    display(_fig)
    _fig
end

using KernelDensity

function kdeplot!(k::UnivariateKDE, xmin, xmax; kws...)
    plot!(range(xmin, xmax, length=200), z->pdf(k, z); grid=:none, label="", kws...)
end

function kdeplot!(x; xmin=quantile(x, 0.05), xmax=quantile(x, 0.95), kws...)
    kdeplot!(kde(x), xmin, xmax; kws...)
end

function kdeplot!(x, bw::Float64; xmin=quantile(x, 0.05), xmax=quantile(x, 0.95), kws...)
    kdeplot!(kde(x, bandwidth=bw), xmin, xmax; kws...)
end

using DataFrames

function results_table(results; drop_constant=true)
    tbl = map(results) do res
        mle = load(res, :mle)
        mle = (i=0, n_obs=0, mle...)
        reopt = load(res, :reopt)[1]

        # x = @Select(σ_obs, sample_cost, switch_cost, α, μ)(mle)
        (mle...,
         train_loss=mean(x[1] / x[3] for x in reopt.train_like),
         test_loss=mean(x[1] / x[3] for x in reopt.test_like)
         )
    end |> DataFrame
    delete!(tbl, :i)
    delete!(tbl, :n_obs)
    if drop_constant
        for k in names(tbl)
            length(unique(tbl[:, k])) == 1 && delete!(tbl, k)
        end
    end
    tbl
end
