using Plots
include("addm.jl")
include("plots_features.jl")
gr(label="", size=(450,400), grid=:none)
Plots.scalefontsizes()
Plots.scalefontsizes(1.2)
flatten = SplitApplyCombine.flatten

# %% ==================== Plotting abstractions ====================

function cross!(x, y; kws...)
    vline!([x], line=(:grey, 0.7), label=""; kws...)
    hline!([y], line=(:grey, 0.7), label=""; kws...)
end

function collapse_subjects(f, trials)
    ys = map(f, collect(group(t->t.subject, trials)))
    map(invert(ys)) do y
        mean(filter(!isnan, y))
    end
end

function make_plot_data(f, bins)
    binf = binnify(f, bins)
    hx, hy = mids(bins), collapse_subjects(binf, TRIALS)
    mx, my = mids(bins), binf(SIM)
    mx, my, hx, hy
end

OPEN_PLOT = false
function save(name)
    savefig("$out/$name.pdf")
    OPEN_PLOT && run(`open $out/$name.pdf`)
    println("Wrote $out/$name.pdf")
end

function binnify(f, bins)
    function binned(trials)
        x, y = f(trials)
        mean.(bin_by(bins, x, y))
    end
end