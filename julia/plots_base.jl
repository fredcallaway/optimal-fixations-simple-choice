using Memoize

fit_mode = "joint"
fit_prior = "false"

if !@isdefined(plot_human!)  # don't reload every time
    include("plots_features.jl")
    include("human.jl")
    using Serialization
    using StatsPlots
    using Printf
    using Bootstrap
    using KernelDensity
    using DataFrames
    using Glob
    using StatsBase
    pyplot(label="")
    Plots.scalefontsizes()
    Plots.scalefontsizes(1.5)

    both_trials = map(["two", "three"]) do num
        get_fold(load_dataset(num), "odd", :test)
    end

    @memoize function load_sims(run_name, i)
        deserialize("results/$run_name/simulations/$fit_mode-$fit_prior/$i")
    end

    @memoize function get_sim(run_name, n_item, i)
        pol_sims = load_sims(run_name, i)[n_item-1]
        reduce(vcat, pol_sims)
    end

    @memoize function get_full_sim(run_name, n_item)
        mapreduce(vcat, 1:30) do i
            get_sim(run_name, n_item, i)
        end
    end

    @memoize function load_features(run_name, i)
        deserialize("results/$run_name/plot_features/$fit_mode-$fit_prior/$i")
    end

    function get_feature(run_name, n_item, i, feature, feature_kws)
        feats = load_features(run_name, i)
        x, y, err, n = Dict(feats[(feature=feature, feature_kws...)])[n_item]
    end
end

PALETTE = Dict(
    ("main14", true) => colorant"#b00007",
    ("main14", false) => colorant"#ff6167",
    ("lesion19", true) => colorant"#003085",
    ("lesion19", false) => colorant"#699efa",
    ("rando17", true) => colorant"#1ba87e",
    ("rando17", false) => colorant"#5fcfad",
    # ("rando17", true) => colorant"#208020",
    # ("rando17", false) => colorant"#6bd16b",
)
ALPHA = 0.7
FILL_ALPHA = 0.3
# ALPHA = 0.3
# FILL_ALPHA = 0.1

CI = 0.95
N_BOOT = 10000
SKIP_BOOT = false
DISABLE_ALIGN = true
FAST = false

# %% ====================  ====================
function ci_err(y)
    length(y) == 1 && return (0., 0.)
    (FAST || SKIP_BOOT) && return (2sem(y), 2sem(y))
    isempty(y) && return (NaN, NaN)
    method = length(y) <= 10 ? ExactSampling() : BasicSampling(N_BOOT)
    bs = bootstrap(mean, y, method)
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

function plot_human!(feature::Function, trials, bins=nothing, type=:line; kws...)
    hx, hy = feature(trials; kws...)
    bins = make_bins(bins, hx)
    plot_human!(bins, hx, hy, type)
end


function plot_model!(x::Vector{Float64}, y, err, type; run_name, total=false, kws...)
    alpha = total ? 1 : ALPHA
    color = PALETTE[run_name, total]
    linewidth = total ? 2 : 1
    if type == :line
        plot!(x, y,
              ribbon=err,
              alpha=alpha,
              fillalpha=FILL_ALPHA,
              color=color,
              linewidth=linewidth,
              label="";
              kws...)
    elseif type == :discrete
        if err != nothing && all(err[1] .≈ 0)
            err = nothing
        end
        plot!(x, y,
              yerr=err,
              grid=:none,
              color=color,
              alpha=alpha,
              linewidth=linewidth,
              marker=(7, :diamond, color, stroke(0)),
              label="";
              kws...)
    else
        error("Bad plot type : $type")
    end
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

function plot_model_precomputed!(feature, feature_kws, type, n_item, n_sim)
    for run_name in RUN_NAMES
        xs, ys, ns = map(1:n_sim) do i
            x, y, err, n = get_feature(run_name, n_item, i, feature, feature_kws)
            plot_model!(x, y, invert(err), type; run_name=run_name)
            x, y .* n, n
        end |> invert
        x = xs[1]
        y = sum(ys) ./ sum(ns)
        plot_model!(x, y, nothing, type; run_name=run_name, total=true)
    end
end

function cross!(x, y)
    vline!([x], line=(:grey, 0.7), label="")
    hline!([y], line=(:grey, 0.7), label="")
end

# function plot_comparison(feature, trials, sim, bins=nothing, type=:line; kws...)
#     plot()
#     hx, hy = feature(trials; kws...)
#     mx, my = feature(sim; kws...)
#     bins = make_bins(bins, hx)
#     plot_human!(bins, hx, hy, type)
#     plot_model!(bins, mx, my, type)
#     # title!(@sprintf "Loss = %.3f" make_loss(feature, bins)(sim))
# end

function kdeplot!(k::UnivariateKDE, xmin, xmax; kws...)
    plot!(range(xmin, xmax, length=200), z->pdf(k, z); grid=:none, label="", kws...)
end

function kdeplot!(x; xmin=quantile(x, 0.05), xmax=quantile(x, 0.95), kws...)
    kdeplot!(kde(x), xmin, xmax; kws...)
end

function kdeplot!(x, bw::Float64; xmin=quantile(x, 0.05), xmax=quantile(x, 0.95), kws...)
    kdeplot!(kde(x, bandwidth=bw), xmin, xmax; kws...)
end


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

function make_plot_kws(run_name, total)
    (alpha = total ? 1 : ALPHA,
     color = PALETTE[run_name, total],
     linewidth = total ? 2 : 1)
end

function plot_one(feature::Function, n_item::Int, xlab, ylab, plot_kws=();
        binning=nothing, type=:line, xline=nothing, yline=nothing, n_sim=30,
        save=false, name=string(feature), manual=false, kws...)

    f = plot(xlabel=xlab, ylabel=ylab; plot_kws...)
    trials = both_trials[n_item-1]

    if manual
        plotter = feature  # more accurate name
        for run_name in RUN_NAMES
            for i in 1:n_sim
                plotter(get_sim(run_name, n_item, i); make_plot_kws(run_name, false)...)
            end
            plotter(get_full_sim(run_name, n_item); make_plot_kws(run_name, true)...)
        end
        plotter(both_trials[n_item-1]; linewidth=2, color=:black, alpha=1)
    else
        plot_model_precomputed!(feature, kws, type, n_item, n_sim)
        hx, hy = feature(trials; kws...)
        bins = make_bins(binning, hx)
        plot_human!(bins, hx, hy, type)
    end

    make_lines!(xline, yline, trials)
    if save
        savefig(f, "$out_path/$name.pdf")
        !INTERACTIVE && println("Wrote $out_path/$name.pdf")
    end
    f
end

function plot_three(feature, xlab, ylab, plot_kws=(); manual=false, kws...)
    plot_kws = (plot_kws..., size=(430,400))
    plot_one(feature, 3, xlab, ylab, plot_kws; save=true, manual=manual, kws...)
end

function plot_both(feature, xlab, ylab, plot_kws=(); yticks=true, align=:default, name=string(feature), kws...)
    xlab1, xlab2 =
        (xlab == :left_rv) ? ("Left rating - right rating", "Left rating - mean other rating") :
        (xlab == :best_rv) ? ("Best rating - worst rating", "Best rating - mean other rating") :
        (xlab == :last_rv) ? ("Last fixated rating - other rating", "Last fixated rating - mean other") :
        (xlab, xlab)

    if !yticks
        plot_kws = (plot_kws..., yticks=[])
        ylab *= "\n"
    end

    f1 = plot_one(feature, 2, xlab1, ylab, plot_kws; kws...)
    f2 = plot_one(feature, 3, xlab2, ylab, plot_kws; kws...)

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
    ff = plot(f1, f2, size=(900,400), margin=3mm)

    if haskey(Dict(kws), :fix_select)
        name *= "_$(kws[:fix_select])"
    end
    savefig(ff, "$out_path/$name.pdf")
    !INTERACTIVE && println("Wrote  $out_path/$name.pdf")
    # return ff
    return ff
end
