using Memoize
using GLM
using RCall
using Suppressor
# ENV["MPLBACKEND"] = "macosx"

if !@isdefined(RELOAD)
    RELOAD = true
end
if RELOAD  # don't reload every time
    println("Clearing cache...")
    include("plots_features.jl")
    include("human.jl")
    RELOAD = false
    using Serialization
    using StatsPlots
    using Printf
    using Bootstrap
    using KernelDensity
    using DataFrames
    using Glob
    using StatsBase
    pyplot(label="", grid=:none)
    S = 1
    Plots.scalefontsizes()
    Plots.scalefontsizes(1.5)
    mm = StatsPlots.mm

    both_trials = map(["two", "three"]) do num
        load_dataset(num, :test)
    end

    @memoize function load_sims(pp, i)
        deserialize("results/$(pp.run_name)/simulations/$(pp.dataset)-$(pp.prior)/$i")
    end

    @memoize get_ind_sim(run_name) = deserialize("results/$run_name/individual/processed/simulations")
    @memoize get_ind_features(run_name) = deserialize("results/$run_name/individual/processed/plot_features")

    @memoize function get_sim(pp, n_item, i, subject)
        if subject != -1
            return get_ind_sim(pp.run_name)[n_item, subject][i]
        end
        # pp.run_name == "addm" && return deserialize("results/addm/sims")[n_item - 1]
        pp.run_name == "addm" && error("get_sim for addm")
        pol_sims = load_sims(pp, i)[n_item-1]
        reduce(vcat, pol_sims)
    end

    @memoize function get_full_sim(pp, n_item, subject)
        pp.run_name == "addm" && return deserialize("results/addm/sims")[n_item - 1]
        mapreduce(vcat, pp.n_sim) do i
            get_sim(pp, n_item, i, subject)
        end
    end

    @memoize function load_features(pp, i)
        if pp.run_name == "addm"
            deserialize("results/addm/plot_features")
        else
            deserialize("results/$(pp.run_name)/plot_features/$(pp.dataset)-$(pp.prior)/$i")
        end
    end

    function get_feature(pp, n_item, i, subject, feature, feature_kws)
        if subject == -1
            feats = load_features(pp, i)
            x, y, err, n = Dict(feats[(feature=feature, feature_kws...)])[n_item]
        else
            pf = get_ind_features(pp.run_name)
            pf[n_item, subject][i][(feature=feature, feature_kws...)]
        end
    end
end

PALETTE = Dict(
    ("revision", true) => colorant"#b00007",
    ("revision", false) => colorant"#ff6167",
    ("main14", true) => colorant"#b00007",
    ("main14", false) => colorant"#ff6167",
    ("lesion19", true) => colorant"#003085",
    ("lesion19", false) => colorant"#699efa",
    ("sobol18", true) => colorant"#003085",
    ("sobol18", false) => colorant"#699efa",
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
LINE_WIDTH = 2

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

function expand(a, b; amt=.05)
    d = b - a
    (a - amt * d, b + amt * d)
end

function plot_human!(bins, x, y, type=:line; indiv=false, kws...)
    vals = bin_by(bins, x, y)
    err = ci_err.(vals)
    bx = mids(bins)
    by = mean.(vals)
    if type == :line
        if indiv
            scatter!(bx, by, alpha=0.4,
                  marker=(3, :circle, stroke(0)),
                  color=:black)
            scatter!(bx, by, yerr=err, alpha=0.3,
                  marker=(3, :circle, 0.0),
                  color=:black)
        else
            plot!(bx, by, yerr=err,
                  line=(LINE_WIDTH,),
                  color=:black,
                  label="";
                  kws...)
        end
    elseif type == :discrete
        Plots.bar!(bx, by,
            yerr=err,
              fill=:white,
              fillalpha=0,
              line=(LINE_WIDTH,),
              color=:black,
              label="";
              kws...)
    else
        error("Bad plot type : $type")
    end
    return bx, by
end



function plot_model!(x::Vector{Float64}, y, err, type, pp; total=false, alpha_mult=1, kws...)
    if type == :individual
        scatter!(x, y,
              yerr=err,
              marker=(3, :diamond, stroke(0))
              ; make_plot_kws(pp, total, alpha_mult)...,
              kws...)
        # plot!(x, y,
        #      marker=(3, :diamond, stroke(0)),
        #      # err=err,
        #      line=nothing,
        #      make_plot_kws(pp, total, 0.3)...,
        #      kws...
        #     )
    elseif type == :line
        plot!(x, y,
              ribbon=err,
              fillalpha=alpha_mult * FILL_ALPHA * pp.alpha,
              label="";
              make_plot_kws(pp, total, alpha_mult)...,
              kws...)
    elseif type == :discrete
        if err != nothing && all(err[1] .≈ 0)
            err = nothing
        end
        plot!(x, y,
              yerr=err,
              marker=(7, :diamond, stroke(0)),
              label="";
              make_plot_kws(pp, total, alpha_mult)...,
              kws...)
    else
        error("Bad plot type : $type")
    end
end

function plot_model!(bins, x, y, pp, type=:line; kws...)
    vals = bin_by(bins, x, y)
    err = invert(ci_err.(vals))
    x = mids(bins)
    y = mean.(vals)
    # TODO
    # too_few = length.(vals) .< 30
    # if !all(length.(vals) .== 1)
    #     x[too_few] .= NaN; y[too_few] .= NaN
    # end
    plot_model!(x, y, err, type, pp; kws...)
end

function plot_model_precomputed!(feature, feature_kws, type, n_item, subject)
    if subject != -1
        type = :individual
    end
    for pp in PARAMS
        if pp.n_sim == 1
            x, y, err, n = get_feature(pp, n_item, 1, subject, feature, feature_kws)
            plot_model!(x, y, err, type, pp, total=true)
        else
            xs, yns, ns = map(1:pp.n_sim) do i
                x, y, err, n = get_feature(pp, n_item, i, subject, feature, feature_kws)
                pp.total_only || plot_model!(x, y, invert(err), type, pp)
                x, y .* n, n
            end |> invert
            x = xs[1]; @assert x ≈ xs[2]
            y = sum(yns) ./ sum(ns)
            plot_model!(x, y, nothing, type, pp, total=true)
        end
    end
end

function cross!(x, y)
    vline!([x], line=(:grey, 0.2), label="")
    hline!([y], line=(:grey, 0.2), label="")
end


function kdeplot!(k::UnivariateKDE, xmin, xmax; kws...)
    plot!(range(xmin, xmax, length=200), z->pdf(k, z); kws...)
end

function kdeplot!(x; xmin=quantile(x, 0.05), xmax=quantile(x, 0.95), kws...)
    kdeplot!(kde(x), xmin, xmax; kws...)
end

function kdeplot!(x, bw::Float64; xmin=quantile(x, 0.05), xmax=quantile(x, 0.95), kws...)
    kdeplot!(kde(x, bandwidth=bw), xmin, xmax; kws...)
end


function make_lines!(xline, yline, trials)
    if xline != nothing
        vline!([xline], line=(:grey, 0.2))
    end
    if yline != nothing
        if yline == :chance
            # yline = 1 / n_item(trials[1])
            yline = 1 / length(trials[1].value)
        end
        hline!([yline], line=(:grey, 0.2))
    end
end

make_plot_kws(pp, total, alpha_mult=1) = (
    alpha = alpha_mult * (total ? pp.alpha ^ (1/2) : ALPHA * pp.alpha),
    color = pp.color[Int(total)+1],
    linewidth = total ? (pp.total_only ? 1 : 3) : 1,
)

add_intercept(x) = hcat(ones(length(x)), x)

# function get_glm_pred(x, y, xmin, xmax)
#     m = if Set(y) <= Set([true, false])
#         glm(add_intercept(x), y, Binomial())
#     else
#         lm(add_intercept(x), y)
#     end

#     xx = range(xmin, xmax, length=500)
#     xx, predict(m, add_intercept(xx))
# end

function stan_glm(x, y, xmin, xmax)
    xhat = range(xmin, xmax, length=500)
    family = Set(y) <= Set([true, false]) ? "binomial" : "gaussian"
    uy = unique(y)
    if length(uy) == 1
        # breaks stan_glm
        return xhat, uy .* ones(length(xhat)), zeros(length(xhat)), zeros(length(xhat))
    end
    Y = try
        @suppress begin
            R"""
            library('rstanarm')
            mod = stan_glm(y ~ x, data=data.frame(x=$x, y=$y), family=$family)
            posterior_linpred(mod, newdata=data.frame(x=$xhat), transform=TRUE)
            """ |> rcopy
        end;
    catch
        println("Error in rstanarm; trying again")
        serialize("tmp/bad_glm", (x, y))
        begin
            R"""
            library('rstanarm')
            mod = stan_glm(y ~ x, data=data.frame(x=$x, y=$y), family=$family)
            posterior_linpred(mod, newdata=data.frame(x=$xhat), transform=TRUE)
            """ |> rcopy
        end;
    end
    est, lo, hi = map(eachcol(Y)) do y
        yhat = mean(y)
        lo, hi = quantile(y, [0.025, 0.975])
        yhat, yhat - lo, hi - yhat
    end |> invert
    xhat, est, lo, hi
    # xhat, Y
end

function get_glm_pred(x, y, xmin, xmax)
    xhat = range(xmin, xmax, length=500)
    family = Set(y) <= Set([true, false]) ? "binomial" : "gaussian"
    P =  R"""
    mod = glm(y ~ x, data=data.frame(x=$x, y=$y), family=$family)
    p = predict(mod, newdata = data.frame(x=$xhat), type = "link", se.fit = TRUE)
    ilink <- family(mod)$linkinv
    data.frame(est=ilink(p$fit), hi=ilink(p$fit + 2 * p$se.fit), lo=ilink(p$fit - 2 * p$se.fit))
    """ |> rcopy
    xhat, P.est, P.est .- P.lo, P.hi .- P.est
end

function get_bins(feature, n_item, bin_spec, feature_kws)
    trials = load_dataset(n_item, :test)
    hx, hy = feature(trials; feature_kws...)
    bins = make_bins(bin_spec, hx)
end

function plot_one(feature::Function, n_item::Int, xlab, ylab, plot_kws=();
        binning=nothing, type=:line, xline=nothing, yline=nothing, x_max=nothing,
        save=false, name=string(feature), manual=false, plot_model=true, precomputed=true, subject=-1, kws...)

    f = plot(xlabel=xlab, ylabel=ylab; plot_kws...)
    trials = both_trials[n_item-1]
    if subject != -1
        trials = filter(t -> t.subject == subject, trials)
        isempty(trials) && error("Invalid subject")
    end

    if manual
        plotter = feature  # more accurate name
        for pp in PARAMS
            if plot_model
                if pp.n_sim > 1 && !pp.total_only
                    for i in 1:pp.n_sim
                        i > 3 && FAST && break
                        plotter(get_sim(pp, n_item, i, subject); id=pp.id, total=false, make_plot_kws(pp, false)...)
                    end
                end
                plotter(get_full_sim(pp, n_item, subject); id=pp.id, total=true, make_plot_kws(pp, true)...)
            end
        end
        plotter(trials; id=:human, linewidth=2, color=:black, alpha=1)
    else
        precomputed && plot_model && plot_model_precomputed!(feature, kws, type, n_item, subject)

        hx, hy = feature(trials; kws...)
        if x_max != nothing
            hx, hy = clip_x(hx, hy, x_max)
        end
        bins = get_bins(feature, n_item, binning, kws)

        if !precomputed
            for pp in PARAMS
                if plot_model
                    if pp.n_sim > 1
                        for i in 1:pp.n_sim
                            mx, my = feature(get_full_sim(pp, n_item, subject); kws...)
                            plot_model!(bins, mx, my, pp)
                        end
                    end
                    mx, my = feature(get_full_sim(pp, n_item, subject); kws...)
                    plot_model!(bins, mx, my, pp)
                end
            end
        end


        # bins = make_bins(binning, hx)
        if subject == -1
            plot_human!(bins, hx, hy, type, lw=1)
        else  # for individual fits, we plot GLM predictions as well
            pp = only(PARAMS)
            if type == :line
                mx, my = feature(get_full_sim(pp, n_item, subject); kws...)
                if x_max != nothing
                    mx, my = clip_x(mx, my, x_max)
                end
                xhat, est, lo, hi = stan_glm(mx, my, bins.limits[1], bins.limits[end])
                plot!(xhat, est, ribbon=(lo, hi), color=pp.color[2], fillalpha=0.2, lw=2)
                # plot!(xhat, Y'[:, 1:1000], line=(0.03, pp.color[2]))
                # plot!(xhat, mean(Y; dims=1)[:], line=(2, pp.color[2]))
            end
            plot_human!(bins, hx, hy, type, alpha=0.3; indiv=true)
            if type == :line
                xhat, est, lo, hi = stan_glm(hx, hy, bins.limits[1], bins.limits[end])
                plot!(xhat, est, ribbon=(lo, hi), color=:black, fillalpha=0.2, lw=2)
                # plot!(xhat, Y'[:, 1:1000], line=(0.03, :black))
                # plot!(xhat, mean(Y; dims=1)[:], line=(2, :black))
            end
        end
    end

    make_lines!(xline, yline, trials)
    if save
        savefig(f, "$out_path/$name.pdf")
        !INTERACTIVE && println("Wrote $out_path/$name.pdf")
    end
    f
end

function clip_x(x, y, x_max)
    keep = x .<= x_max
    x[keep], y[keep]
end

function plot_three(feature, xlab, ylab, plot_kws=(); manual=false, kws...)
    plot_kws = (plot_kws..., size=(430S, 400S))
    plot_one(feature, 3, xlab, ylab, plot_kws; save=true, manual=manual, kws...)
end

function fmt_xlab(xlab)
    (xlab == :left_rv) ? ("Left rating - right rating", "Left rating - mean other rating") :
    (xlab == :best_rv) ? ("Best rating - worst rating", "Best rating - mean other rating") :
    (xlab == :last_rv) ? ("Last fixated rating - other rating", "Last fixated rating - mean other") :
    (xlab, xlab)
end
fmt_xlab(xlab, n_item) = fmt_xlab(xlab)[n_item-1]

function plot_both(feature, xlab, ylab, plot_kws=(); yticks=true, align=:default, name=string(feature), kws...)
    xlab1, xlab2 = fmt_xlab(xlab)

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
    ff = plot(f1, f2, size=(900S,400S), margin=3mm)

    if haskey(Dict(kws), :fix_select)
        name *= "_$(kws[:fix_select])"
    end
    savefig(ff, "$out_path/$name.pdf")
    !INTERACTIVE && println("Wrote  $out_path/$name.pdf")
    # return ff
    return ff
end
