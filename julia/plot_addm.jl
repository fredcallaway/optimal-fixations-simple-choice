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
sim = simulate_trials(ADDM(), repeat(trials, 10))

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
    bins = Binning(-100:200:700)    
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
fig5d = plot(xlabel="First fixation duration (ms)", ylabel="P(first seen chosen)", ylim=(0, 1))
plot5d!(trials, bar!, fill=:white)
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
plot(fig5a, fig5b, fig5c, fig5d, fig5e, size=(1200,600))
savefig("figs/addm/full.pdf")


# %% --------

function relative_left(x)
    x[1] - sum(x[2:end])
end