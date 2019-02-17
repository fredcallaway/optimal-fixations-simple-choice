include("binning.jl")

function make_bins(bins, hx)
    if bins == :integer
        return Binning(minimum(hx)-0.5:1:maximum(hx)+0.5)
    elseif bins isa Nothing
        bins = 5
    end
    if bins isa Int
        low, high = quantile(hx, [0.05, 0.95])
        bin_size = (high - low) / bins
        bins = Binning(low:bin_size:high)
        # bins = Binning(quantile(hx, 0:1/bins:1))
    end
    return bins
end


# %% ==================== fixation time -> choice ====================

function total_fix_time(t)::Vector{Float64}
    x = zeros(3)
    for (fi, ti) in zip(t.fixations, t.fix_times)
        x[fi] += ti
    end
    return x
end

function fixation_bias(trials)
    mapmany(trials) do t
        ft = total_fix_time(t)
        # invert((ft ./ sum(ft), t.choice .== 1:3))
        invert((ft .- mean(ft), t.choice .== 1:3))
    end |> Vector{Tuple{Float64, Bool}} |> invert
end


# %% ==================== value difference -> time ====================

difficulty(v) = maximum(v) - mean(v)

function difference_time(trials)
    difficulty.(trials.value), sum.(trials.fix_times)
end


# %% ==================== max value -> choice value ====================

choice_value(t) = t.value[t.choice]

function value_choice(trials)
    Int.(maximum.(trials.value)), choice_value.(trials)
end


# %% ==================== First fixation duration -> choose first fixated ====================

function first_fixation_bias(trials)
    map(trials) do t
        t.fix_times[1], t.choice == t.fixations[1]
    end |> invert
end


# %% ==================== Last fixation chosen ====================

choose_last_fixated(t) = t.fixations[end] == t.choice

function last_fix_bias(trials)
    map(trials) do t
        last = t.fixations[end]
        t.value[last] - mean(t.value), t.choice == last
    end |> invert
end


# %% ==================== value -> fixation time ====================

function value_bias(trials)
    mapmany(trials) do t
        tft = total_fix_time(t)
        invert((t.value .- mean(t.value), tft ./ sum(tft)))
    end |> invert
end
