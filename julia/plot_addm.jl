using Plots
include("addm.jl")
include("plots_features.jl")
gr(label="", size=(450,400))
Plots.scalefontsizes()
Plots.scalefontsizes(1.2)

# %% ==================== Load data and simulate ====================

trials = load_dataset(2, :test)
trials = filter(trials) do t
    difficulty(t.value) <= 5
end

MAX_FIX = 20
SIMULATE_TRIALS = false

if SIMULATE_TRIALS
    out = "figs/addm-alt"
    sim = simulate_trials(ADDM(), repeat(trials, 80))
else

    out = "figs/addm"
    vs = filter(collect(Iterators.product(0:10, 0:10))) do v
        abs(v[1] - v[2]) <= 5
    end
    m = ADDM()
    fp = BinaryFixationProcess()
    sim = mapmany(vs) do v
        map(1:1000) do i
            while true
                sim = simulate(m, fp, collect(v))[1]
                length(sim.fixations) <= MAX_FIX && return sim
            end
            sim
        end
    end;
end
mkpath(out)
# %% ==================== Plotting abstractions ====================

function cross!(x, y)
    vline!([x], line=(:grey, 0.7), label="")
    hline!([y], line=(:grey, 0.7), label="")
end

function collapse_subjects(f, trials)
    ys = map(f, collect(group(t->t.subject, trials)))
    map(invert(ys)) do y
        mean(filter(!isnan, y))
    end
end

function make_plot(f, name, bins, plot_human=bar!; kws...)
    binf = binnify(f, bins)
    fig = plot(;kws...)
    plot_human(mids(bins), collapse_subjects(binf, trials); color=:black, fill=:white)
    plot!(mids(bins), binf(sim), color=:red)
    savefig("$out/$name.pdf")
    println("Wrote $out/$name.pdf")
end

function binnify(f, bins)
    function binned(trials)
        x, y = f(trials)
        mean.(bin_by(bins, x, y))
    end
end

# %% ==================== 4b ====================
kws = (xlabel="Last seen item rating - other item rating", ylabel="P(last fixation to chosen)", ylim=(0,1))
make_plot(last_fix_bias, "4b", Binning(-5.5:1:5.5), plot!; kws...)

# %% ==================== 4c ====================
kws = (xlabel="Last fixation duration (ms)", ylabel="Time advantage (ms)")
make_plot("4c", Binning(0:400:1600)) do trials
    advantage, last_duration = last_fixation_duration(trials)
    last_duration, advantage
end


# %% ==================== 5a ====================

bins = Binning(-5.5:5.5)
binf = binnify(value_choice, bins)

splitlast(trials) = group(trials) do t
    t.fixations[end] == 1
end
H = map(splitlast(trials)) do tt
    collapse_subjects(binf, tt)
end
M = map(binf, splitlast(sim))


plot(xlabel="Left rating - right rating", ylabel="P(left chosen)", ylim=(0,1))
cross!(0, 0.5)

for t in [true, false]
    plot!(mids(bins), M[t], color=:red, marker=(7, :circle, :white, stroke(:red)))
    plot!(mids(bins), H[t], color=:black, marker=(7, :circle, :white, stroke(:black)))
end
savefig("$out/5a.pdf")

# %% ==================== 5b ====================
bins = make_bins(nothing, x)
kws = (xlabel="Final time advantage left (ms)", ylabel="P(left chosen)", ylim=(0,1))
make_plot("5b", Binning(-700:200:700); kws...) do trials
    fixation_bias(trials)
end


# %% ==================== 5d ====================

kws = (xlabel="First fixation duration (ms)", ylabel="P(first seen chosen)", ylim=(0, 1))
make_plot("5d", Binning(0:200:800); kws...) do trials
    x, y = Float64[], Bool[]
    for t in trials
        if length(t.fixations) > 0
            push!(x, t.fix_times[1])
            push!(y, t.choice == t.fixations[1])
        end
    end
    x, y
end

# %% --------

function plot5e!(trials, plot=plot!; kws...)
    function firstval(t)
        f1 = t.fixations[1]
        f2 = [2, 1][f1]
        t.value[f1] - t.value[f2]
    end
    G = group(firstval, trials)
    p_first = map(G) do trials
        mean(t.choice == t.fixations[1] for t in trials)
    end

    bins = Binning(0:200:800)
    x, y = Float64[], Float64[]
    for t in trials
        if length(t.fixations) > 0
            push!(x, t.fix_times[1])
            push!(y, (t.choice == t.fixations[1]) - p_first[firstval(t)])
        end
    end
    biny = mean.(bin_by(bins, x, y))
    plot(mids(bins), biny; kws...)
end
fig5e = plot(xlabel="First fixation duration (ms)", ylabel="Corrected P(first seen chosen)", ylim=(-0.3, 0.3))
plot5e!(trials, bar!, fill=:white)
plot5e!(sim, color=:red)
savefig("$out/5e.pdf")



# %% ==================== Accumulation ====================

fp = TrinaryFixationProcess()
t, h = simulate(ADDM(θ=0, σ=0.005), fp, [9,9,9]; save_history=true)
H = reduce(hcat, h)'
plot(H, lab="")
vline!(cumsum(t.fix_times), c=:black)


# %% ==================== Random shit ====================


# %% --------
# FIXATION DURATIONS
x, y = binned_fixation_times(trials)
bins = make_bins(:integer, x)
biny = mean.(bin_by(bins, x, y))

# %% --------
mapmany(trials) do t
    t.fix_times[2:end-1]
end |> mean

# %% --------
dc = countmap(difficulty.(trials.value))

fp = BinaryFixationProcess()
# %% --------
valmap(mean, fp.other_durations) |> sort |> values |> collect
map(0:8) do d
    mapmany(1:1000) do i
        sim = simulate(ADDM(), fp, [v1, v1 + d])[1]
        sim.fix_times[2:end-1]
    end |> mean
end

# %% --------
plot(fig5a, fig5b, fig5c, fig5d, fig5e, size=(1200,600))
savefig("$out/full.pdf")

# %% --------
trials.fixations |> flatten |> length
map(trials) do t
    t.value[t.choice]
end |> mean
mapmany(trials) do t
    t.fix_times
end |> mean

length(trials)

# %% --------



function plot5c!(trials, plot=plot!; kws...)
    G = group(trials) do t
        t.value[1] - t.value[2]
    end
    p_left = map(G) do trials
        2 - mean(getfield.(trials, :choice))
    end
    bins = Binning(-700:200:700)

    x = Float64[]; y = Float64[];
    for t in trials
        push!(x, relative_left(total_fix_times(t)))
        push!(y, (t.choice == 1) - p_left[t.value[1] - t.value[2]])
    end
    
    biny = mean.(bin_by(bins, x, y))
    plot(mids(bins), biny; kws...)
end

fig5c = plot(xlabel="Final time advantage left (ms)", ylabel="Corrected P(left chosen)", ylim=(-0.3, 0.3))
plot5c!(trials, bar!, fill=:white)
plot5c!(sim, color=:red)
savefig("$out/5c.pdf")
