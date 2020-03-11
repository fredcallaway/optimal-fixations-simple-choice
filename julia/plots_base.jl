ENV["GKSwstype"] = "nul"

include("features.jl")
using Serialization
using StatsPlots
pyplot(label="")
plot([1,2]);


# %% ====================  ====================
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
using Printf

RED = colorant"#FF6167"
NO_RIBBON = false
FAST = false
SKIP_BOOT = false
DISPLAY = false
const CI = 0.95
const N_BOOT = 10000
using Bootstrap
function ci_err(y)
    length(y) == 1 && return (0., 0.)
    NO_RIBBON && return (0., 0.)
    (FAST || SKIP_BOOT) && return (sem(y) * 2, sem(y) * 2)
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
              fillalpha=0,
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
    x = mids(bins)
    y = mean.(vals)
    too_few = length.(vals) .< 30
    if !all(length.(vals) .== 1)
        x[too_few] .= NaN; y[too_few] .= NaN
    end
    plot_model!(x, y, err, type; kws...)
end

function plot_model!(x::Vector{Float64}, y, err, type; color=RED, kws...)
    if type == :line
        plot!(x, y,
              ribbon=err,
              fillalpha=0.05,
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

# function fig(f, name)
#     _fig = f()
#     savefig("figs/$run_name/$name.pdf")
#     display(_fig)
#     _fig
# end

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

function make_lines!(xline, yline, trials)
    if xline != nothing
        vline!([xline], line=(:grey, 0.7))
    end
    if yline != nothing
        if yline == :chance
            # yline = 1 / n_item(trials[1])
            yline = 1 / length(trials[1].value)
        end
        hline!([yline], line=(:grey, 0.7))
    end
end


function plot_one(feature, xlab, ylab, trials, sims, plot_kws=();
        binning=nothing, type=:line, xline=nothing, yline=nothing,
        save=false, name=string(feature), kws...)
    hx, hy = feature(trials; kws...)
    bins = make_bins(binning, hx)
    f = plot(xlabel=xlab, ylabel=ylab; plot_kws...)
    # plot_human!(bins, hx, hy, type)

    if FAST
        sims = sims[1:2]
    end
    sims = reverse(sims)
    colors = range(colorant"red", stop=colorant"blue",length=length(sims))
    for (c, sim) in zip(colors, sims)
        mx, my = feature(sim; kws...)
        plot_model!(bins, mx, my, type, alpha=0.2) # color=c
    end
    plot_human!(bins, hx, hy, type)

    make_lines!(xline, yline, trials)
    if save
        savefig(f, "$out_path/$name.pdf")
    end
    f
end

function plot_one(name::String, xlab, ylab, trials, sims, plot_kws;
        xline=nothing, yline=nothing,
        plot_human::Function, plot_model::Function)

    f = plot(xlabel=xlab, ylabel=ylab; plot_kws...)

    # plot_human(trials)

    if FAST
        sims = sims[1:4]
    end
    sims = reverse(sims)
    colors = range(colorant"red", stop=colorant"blue",length=length(sims))
    for (c, sim) in zip(colors, sims)
        plot_model(sim) # color=c
    end
    plot_human(trials)

    make_lines!(xline, yline, trials)
    f
end

DISABLE_ALIGN = true
function plot_both(feature, xlab, ylab, plot_kws=(); yticks=true, align=:default, name=string(feature), kws...)
    xlab1, xlab2 =
        (xlab == :left_rv) ? ("Left rating - right rating", "Left rating - mean other rating") :
        (xlab == :best_rv) ? ("Best rating - worst rating", "Best rating - mean other rating") :
        (xlab == :last_rv) ? ("Last fixated rating - other rating", "Last fixated rating - mean other") :
        (xlab, xlab)

    # ff = plot_one(feature, xlab, ylab, trials, sims, plot_kws; kws...)
    # if haskey(Dict(kws), :fix_select)
    #     name *= "_$(kws[:fix_select])"
    # end
    # savefig(ff, "$out_path/$name.pdf")
    # return ff
    if !yticks
        plot_kws = (plot_kws..., yticks=[])
        ylab *= "\n"
    end

    f1 = plot_one(feature, xlab1, ylab, both_trials[1], both_sims[1], plot_kws; kws...)
    f2 = plot_one(feature, xlab2, ylab, both_trials[2], both_sims[2], plot_kws; kws...)

    ylabel!(f2, yticks ? "  " : " \n ")
    x1 = xlims(f1); x2 = xlims(f2)
    y1 = ylims(f1); y2 = ylims(f2)
    for (i, f) in enumerate([f1, f2])
        if align != :y_only
            xlims!(f, min(x1[1], x2[1]), max(x1[2], x2[2]))
        end
        if align == :default || DISABLE_ALIGN
            ylims!(f, min(y1[1], y2[1]), max(y1[2], y2[2]))
        elseif align == :chance
            rng = max(maximum(abs.(y1 .- 1/2)), maximum(abs.(y2 .- 1/3)))
            chance = [1/2, 1/3][i]
            ylims!(f, chance - rng, chance + rng)
        end
    end
    ff = plot(f1, f2, size=(900,400))

    if haskey(Dict(kws), :fix_select)
        name *= "_$(kws[:fix_select])"
    end
    savefig(ff, "$out_path/$name.pdf")
    # return ff
    DISPLAY && display(ff)
    return
end
