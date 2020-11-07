using Plots
include("addm.jl")
include("plots_features.jl")
gr(label="", size=(450,400))
Plots.scalefontsizes()
Plots.scalefontsizes(1.2)

function cross!(x, y)
    vline!([x], line=(:grey, 0.7), label="")
    hline!([y], line=(:grey, 0.7), label="")
end
# %% ==================== Accumulation ====================

fp = TrinaryFixationProcess()
t, h = simulate(ADDM(θ=0, σ=0.005), fp, [9,9,9]; save_history=true)
H = reduce(hcat, h)'
plot(H, lab="")
vline!(cumsum(t.fix_times), c=:black)

# %% ==================== KAR 2010 Figure 5 ====================

trials = load_dataset(2, :test)
trials = filter(trials) do t
    difficulty(t.value) <= 5
end

# sim = simulate_trials(ADDM(), repeat(trials, 10))
# %% --------
vs = filter(collect(Iterators.product(0:10, 0:10))) do v
    abs(v[1] - v[2]) <= 5
end

MAX_FIX = 20

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

# %% --------

function plot5a!(trials; kws...)
    bins = Binning(-5.5:5.5)
    G = group(trials) do t
        t.fixations[end] == 1
    end
    y = map(G) do trials
        mean.(bin_by(bins, value_choice(trials)...))
    end
    x = mids(bins)
    plot!(x, y[true]; kws...)
    plot!(x, y[false]; kws...)
end

fig5a = plot(xlabel="Left rating - right rating", ylabel="P(left chosen)", ylim=(0,1))
plot5a!(trials; color=:black)
plot5a!(sim; color=:red)
cross!(0, 0.5)
savefig("figs/addm/5a.pdf")
# %% --------

function plot5b!(trials, plot=plot!; kws...)
    # bins = Binning(-600:200:600)
    bins = Binning(-700:200:700)
    y = mean.(bin_by(bins, fixation_bias(trials)...))
    plot(mids(bins), y; kws...)
end

fig5b = plot(xlabel="Final time advantage left (ms)", ylabel="P(left chosen)", ylim=(0,1))
plot5b!(trials, bar!, fill=:white)
plot5b!(sim; color=:red)
savefig("figs/addm/5b.pdf")
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
savefig("figs/addm/5c.pdf")
# %% --------

function plot5d!(trials, plot=plot!; kws...)
    bins = Binning(0:200:800)
    x, y = Float64[], Bool[]
    for t in trials
        if length(t.fixations) > 0
            push!(x, t.fix_times[1])
            push!(y, t.choice == t.fixations[1])
        end
    end
    biny = mean.(bin_by(bins, x, y))
    plot(mids(bins), biny; kws...)
end

function alt_plot5d!(trials, plot=plot!; kws...)
    bins = Binning(0:200:800)
    biny = map(collect(group(t->t.subject, trials))) do tt
        x, y = Float64[], Bool[]
        for t in tt
            if length(t.fixations) > 0
                push!(x, t.fix_times[1])
                push!(y, t.choice == t.fixations[1])
            end
        end
       mean.(bin_by(bins, x, y))
    end
    map(invert(biny)) do yy
        mean(filter(!isnan, yy))
    end
    plot(mids(bins), biny; kws...)
end

fig5d = plot(xlabel="First fixation duration (ms)", ylabel="P(first seen chosen)", ylim=(0, 1))
alt_plot5d!(trials, bar!, fill=:white)
plot5d!(sim, color=:red)
savefig("figs/addm/5d.pdf")
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
savefig("figs/addm/5e.pdf")

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
savefig("figs/addm/full.pdf")

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

