include("plot_addm_base.jl")

TRIALS = filter(load_dataset(2, :test)) do t
    difficulty(t.value) <= 5
end

SIMULATE_TRIALS = false
# %% --------
if SIMULATE_TRIALS
    out = "figs/addm/binary-alt"
    SIM = simulate_trials(ADDM(), repeat(TRIALS, 80))
else
    out = "figs/addm/binary"
    vs = filter(collect(Iterators.product(0:10, 0:10))) do v
        abs(v[1] - v[2]) <= 5
    end
    m = ADDM()
    fp = BinaryFixationProcess()
    SIM = mapmany(vs) do v
        map(1:1000) do i
            simulate(m, fp, collect(v))[1]
        end
    end;
end
mkpath(out)

# %% ==================== 4b ====================

plot(xlabel="Last seen item rating - other item rating", ylabel="P(last fixation to chosen)", ylim=(0,1))
mx, my, hx, hy = make_plot_data(last_fix_bias, Binning(-5.5:1:5.5))

scatter!(hx, hy, marker=(7, :circle, :white, stroke(:black)))
plot!(mx, my, ls=:dash, color=:red)
i = argmin(abs.(my .- 1/2))
cross!(mx[i], 1/2, ls=:dot)
save("4b")


# %% ==================== 4c ====================
plot(xlabel="Last fixation duration (ms)", ylabel="Time advantage (ms)", 
     ylim=(-850, 50), xticks=(0:400:1600))

mx, my, hx, hy = make_plot_data(Binning(0:400:1600)) do trials
    last_duration = Float64[]; advantage = Float64[]
    for t in trials
        length(t.fixations) == 0 && continue
        last = t.fixations[end]
        # last != t.choice && continue
        tft = total_fix_times(t)
        tft[last] -= t.fix_times[end]
        adv = 2 * tft[t.choice] - sum(tft)
        # adv = tft[t.choice] - mean(tft)
        if t.fix_times[end] < 1500
            push!(advantage, adv)
            push!(last_duration, t.fix_times[end])
        end
    end
    last_duration, advantage
end
bar!(hx, hy; color=:black, fill=:white)
plot!(mx, my, color=:red, ls=:dash, marker=(7, :circle, :white, stroke(:red)))
save("4c")

# # %% ==================== 5a ====================

# bins = Binning(-5.5:5.5)
# binf = binnify(value_choice, bins)

# splitlast(trials) = group(trials) do t
#     t.fixations[end] == 1
# end
# H = map(splitlast(trials)) do tt
#     collapse_subjects(binf, tt)
# end
# M = map(binf, splitlast(sim))


# plot(xlabel="Left rating - right rating", ylabel="P(left chosen)", ylim=(0,1))
# cross!(0, 0.5)

# for t in [true, false]
#     plot!(mids(bins), M[t], color=:red, marker=(7, :circle, :white, stroke(:red)))
#     plot!(mids(bins), H[t], color=:black, marker=(7, :circle, :white, stroke(:black)))
# end
# savefig("$out/5a.pdf")

# %% ==================== 5b ====================

plot(xlabel="Final time advantage left (ms)", ylabel="P(left chosen)", ylim=(0,1))
mx, my, hx, hy = make_plot_data(Binning(-700:200:700)) do trials
    fixation_bias(trials)
end
bar!(hx, hy; color=:black, fill=:white)
plot!(mx, my, color=:red, ls=:dash, marker=(7, :circle, :white, stroke(:red)))
save("5b")


# %% ==================== 5d ====================

plot(xlabel="First fixation duration (ms)", ylabel="P(first seen chosen)", ylim=(0, 1))
mx, my, hx, hy = make_plot_data(Binning(0:200:800)) do trials
    x, y = Float64[], Bool[]
    for t in trials
        if length(t.fixations) > 0
            push!(x, t.fix_times[1])
            push!(y, t.choice == t.fixations[1])
        end
    end
    x, y
end

bar!(hx, hy; color=:black, fill=:white)
plot!(mx, my, color=:red, ls=:dash, marker=(7, :circle, :white, stroke(:red)))
save("5d")