include("binning.jl")

function make_bins(bins, hx)
    if bins == :integer
        return Binning(minimum(hx)-0.5:1:maximum(hx)+0.5)
    elseif bins isa Nothing
        bins = 7
    end
    if bins isa Int
        low, high = quantile(hx, [0.1, 0.9])
        bin_size = (high - low) / bins
        bins = Binning(low:bin_size:high)
        # bins = Binning(quantile(hx, 0:1/bins:1))
    end
    return bins
end

function total_fix_time(t)::Vector{Float64}
    x = zeros(3)
    for (fi, ti) in zip(t.fixations, t.fix_times)
        x[fi] += ti
    end
    return x
end

function fixation_times(trials)
    x = Tuple{Int, Float64}[]
    for t in trials
        f = t.fix_times
        push!(x, (1, f[1]))
        length(f) > 1 && push!(x, (2, f[2]))
        for i in 3:length(f)-1
            push!(x, (3, f[i]))
        end
        push!(x, (4, f[end]))
    end
    invert(x)
end

function fixation_bias(trials)
    mapmany(trials) do t
        ft = total_fix_time(t)
        # invert((ft ./ sum(ft), t.choice .== 1:3))
        invert((ft .- mean(ft), t.choice .== 1:3))
    end |> Vector{Tuple{Float64, Bool}} |> invert
end

difficulty(v) = maximum(v) - mean(v)

function difference_time(trials)
    difficulty.(trials.value), sum.(trials.fix_times)
end

function difference_nfix(trials)
    difficulty.(trials.value), length.(trials.fixations)
end


choice_value(t) = t.value[t.choice]

function old_value_choice(trials)
    Int.(maximum.(trials.value)), choice_value.(trials)
end


function value_choice(trials)
    mapmany(trials) do t
        # invert((ft ./ sum(ft), t.choice .== 1:3))
        invert((t.value .- mean(t.value), t.choice .== 1:3))
    end |> Vector{Tuple{Float64, Bool}} |> invert
end

# %% ==================== First fixation duration -> choose first fixated ====================

function first_fixation_duration(trials)
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

function gaze_cascade(trials; k=6)
    denom = zeros(Int, k)
    num = zeros(Int, k)
    for t in trials
        x = reverse!(t.fixations .== t.choice)
        for i in 1:min(k, length(x))
            denom[i] += 1
            num[i] += x[i]
        end
    end
    -k+1:0, reverse!(num ./ denom)
end

function fixation_value(trials; sample_time=10)
    max_time = 3970  # 90% quanitle of human rounded to nearest 10
    k = ceil(Int, max_time / sample_time)
    denom = zeros(k)
    num = zeros(k)
    for t in trials
        x = t.value[discretize_fixations(t; sample_time=sample_time)] .- mean(t.value)
        for i in 1:min(k, length(x))
            denom[i] += 1
            num[i] += x[i]
        end
    end
    collect(1:k) .* sample_time, num ./ denom
end

function fixate_on_best(trials; sample_time=10, cutoff=CUTOFF)
    # kind = [Tuple(diff(sort(v))) for v in trials.value]
    # trials = trials[kind .== [(1., 1.)]]
    tft = sum.(trials.fix_times)
    trials = trials[cutoff .< tft]
    k = ceil(Int, CUTOFF / sample_time)
    denom = zeros(k)
    num = zeros(k)
    for t in trials
        fix = discretize_fixations(t; sample_time=sample_time)
        # x = t.value[fix] .- mean(t.value)
        x = fix .== argmax(t.value)
        for i in 1:min(k, length(x))
            denom[i] += 1
            num[i] += x[i]
        end
    end
    collect(1:k) .* sample_time, num ./ denom
end

unique_values(t) = length(unique(t.value)) == length(t.value)

function fourth_rank(trials)
    x = Int[]
    for t in trials
        if length(t.fixations) > 3 && sort(t.fixations[1:3]) == 1:3 && unique_values(t)
            ranks = sortperm(sortperm(-t.value))
            push!(x, ranks[t.fixations[4]])
        end
    end
    if length(x) == 0
        return missing
    end
    cx = counts(x, 3)
    1:3, cx / sum(cx)
end

function last_fixation_duration(trials)
    map(trials) do t
        last = t.fixations[end]
        last != t.choice && return missing
        tft = total_fix_time(t)
        tft[last] -= t.fix_times[end]
        adv = 2 * tft[t.choice] - sum(tft)
        # adv = tft[t.choice] - mean(tft)
        (adv, t.fix_times[end])
    end |> skipmissing |> collect |> invert
end
