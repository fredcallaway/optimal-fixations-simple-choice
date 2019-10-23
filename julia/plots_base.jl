rusing Distributed
nprocs() == 1 && addprocs()
@everywhere begin
    cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
    include("model_base.jl")
    include("dc.jl")
end
include("features.jl")
using Serialization
using StatsPlots
plot([1,2])

# %% ====================  ====================
pyplot(label="")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
const N_BOOT = 1000
using Bootstrap
using Printf
using Plots: px
estimator = mean
ci = 0.95
RED = colorant"#FF6167"

function ci_err(estimator, y)
    return sem(y) * 2
    bs = bootstrap(estimator, y, BalancedSampling(N_BOOT))
    c = confint(bs, BasicConfInt(ci))[1]
    abs.(c[2:3] .- c[1])
end

function plot_human!(bins, x, y, type=:line; kws...)
    vals = bin_by(bins, x, y)
    if type == :line
        plot!(mids(bins), estimator.(vals), yerr=ci_err.(estimator, vals),
              grid=:none,
              line=(2,),
              color=:black,
              label="";
              kws...)
    elseif type == :discrete
        Plots.bar!(mids(bins), estimator.(vals),
            yerr=ci_err.(estimator, vals),
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
    if type == :line
        plot!(mids(bins), estimator.(vals),
              ribbon=ci_err.(estimator, vals),
              fillalpha=0.1,
              color=RED,
              line=(RED, 1),
              label="";
              kws...)
    elseif type == :discrete
        plot!(mids(bins), estimator.(vals),
              # yerr=ci_err.(estimator, vals),
              grid=:none,
              line=(RED, 1),
              marker=(7, :diamond, RED, stroke(0)),
              label="";
              kws...)
    else
        error("Bad plot type : $type")
    end
end

function plot_model!(sim, feature::Function, bins=nothing, type=:line; kws...)
    @assert false
    hx, hy = feature(sim)
    bins = make_bins(bins, hx)
    plot_human!(bins, hx, hy, type; kws...)
end

function cross!(x, y)
    vline!([x], line=(:grey, 0.7), label="")
    hline!([y], line=(:grey, 0.7), label="")
end

function plot_comparison(feature, sim, bins=nothing, type=:line; kws...)
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
