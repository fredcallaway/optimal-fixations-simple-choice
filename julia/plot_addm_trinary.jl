include("plot_addm_base.jl")
using CSV

TRIALS = load_dataset(3, :full)

MAX_FIX = 20
SIMULATE_TRIALS = false

if SIMULATE_TRIALS
    out = "figs/addm/trinary-alt"
    SIM = simulate_trials(ADDM(), repeat(TRIALS, 80))
else
    out = "figs/addm/trinary"
    # vs = filter(collect(Iterators.product(1:10, 1:10, 1:10))) do v
    #     length(unique(v)) == 3
    # end
    vs = let
        # Get the values actually used for simulation, extracted from ET3_sim_msprt_excite_e.R
        rsim = CSV.read("/Users/fred/Projects/choice-eye-tracking/julia/addm/trinary/simfirstfix.csv")
        eachrow(convert(Matrix, rsim[[:leftrating, :middlerating, :rightrating]]))
    end
    m = ADDM()
    fp = TrinaryFixationProcess()
    SIM = mapmany(vs) do v
        map(1:500) do i
            while true
                sim = simulate(m, fp, collect(v))[1]
                length(sim.fixations) <= MAX_FIX && return sim
            end
            sim
        end
    end;
end
mkpath(out)

# %% ==================== 3a ====================

fig = plot(;xlabel="Last seen item rating - avg other items", ylabel="P(last fixation to chosen)", 
            ylim=(0,1), xticks=-4:2:6)
mx, my, hx, hy = make_plot_data(last_fix_bias, Binning(-4.25:0.5:7))
scatter!(hx, hy, marker=(7, :circle, :white, stroke(:black)))
plot!(mx, my, ls=:dash, color=:red)
i = argmin(abs.(my .- 1/3))
cross!(mx[i], 1/3, ls=:dot)
save("3a")

# # %% ==================== 3b ====================

plot(xlabel="Last fixation duration (ms)", ylabel="Time advantage (ms)", 
    ylim=(-1250, 50), yticks=-1200:400:0, xticks=0:400:1200)
mx, my, hx, hy = make_plot_data(Binning(0:400:1600)) do trials
    last_duration = Float64[]; advantage = Float64[]
    for t in trials
        length(t.fixations) == 0 && continue
        last = t.fixations[end]
        tft = total_fix_times(t)
        tft[last] -= t.fix_times[end]

        adv = 2tft[t.choice] - sum(tft)
        if t.fix_times[end] < 1500
            push!(advantage, adv)
            push!(last_duration, t.fix_times[end])
        end
    end
    last_duration, advantage
end

bar!(hx, hy; color=:black, fill=:white)
plot!(mx, my, color=:red, ls=:dash, marker=(7, :circle, :white, stroke(:red)))
save("3b")

# %% ==================== 3c ====================

plot(xlabel="Final time advantage left (ms)", ylabel="P(left chosen)", ylim=(0,1))
# make_plot_data(Binning(-700:200:700); kws...) do trials
mx, my, hx, hy = make_plot_data(Binning(-700:200:700)) do trials
    x = Float64[]; y = Bool[];
    for t in trials
        tft = total_fix_times(t)
        push!(x, tft[1] - sum(tft[2:3]))
        push!(y, t.choice == 1)
    end
    x, y
end

bar!(hx, hy; color=:black, fill=:white)
plot!(mx, my, color=:red, ls=:dash, marker=(7, :circle, :white, stroke(:red)))
save("3c")

# %% ==================== 3e ====================

plot(xlabel="First fixation duration (ms)", ylabel="P(first seen chosen)", ylim=(0, 1))
mx, my, hx, hy = make_plot_data(Binning(0:200:800)) do trials
# mx, my, hx, hy = make_plot_data(Binning([0; 100:200:700])) do trials
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
save("3e")
