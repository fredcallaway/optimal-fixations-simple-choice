include("binning.jl")
using StatsBase
const CUTOFF = 2000

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
        length(f) == 0 && continue
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
    x, y = Float64[], Bool[]
    for t in trials
        if length(t.fixations) > 0
            push!(x, t.fix_times[1])
            push!(y, t.choice == t.fixations[1])
        end
    end
    x, y
end


# %% ==================== Last fixation chosen ====================

choose_last_fixated(t) = t.fixations[end] == t.choice

function last_fix_bias(trials)
    x, y = Float64[], Bool[]
    for t in trials
        if length(t.fixations) > 0
            last = t.fixations[end]
            push!(x, t.value[last] - mean(t.value))
            push!(y, t.choice == last)
        end
    end
    x, y
end


# %% ==================== value -> fixation time ====================

function value_bias(trials)
    mapmany(trials) do t
        tft = total_fix_time(t)
        invert((t.value .- mean(t.value), tft ./ (sum(tft) + eps())))
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

# %% ====================  ====================
# CUTOFF = Int(round(quantile(sum.(trials.fix_times), 0.5)))

function fixate_on_best(trials; sample_time=10, cutoff=CUTOFF)
    # kind = [Tuple(diff(sort(v))) for v in trials.value]
    # trials = trials[kind .== [(1., 1.)]]
    rt = sum.(trials.fix_times)
    # trials = trials[cutoff .> tft]
    k = ceil(Int, cutoff / sample_time)
    denom = zeros(k)
    num = zeros(k)
    for t in trials
        # t.choice == argmax(t.value) || continue
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

# %% ====================  ====================

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
    n = length(x)
    p = counts(x, 3) ./ n
    std_ = @. √(p * (1 - p) / n)
    1:3, p, std_
end


function last_fixation_duration(trials)
    x, y = Float64[], Float64[]
    for t in trials
        length(t.fixations) == 0 && continue
        last = t.fixations[end]
        last != t.choice && continue
        tft = total_fix_time(t)
        tft[last] -= t.fix_times[end]
        adv = 2 * tft[t.choice] - sum(tft)
        # adv = tft[t.choice] - mean(tft)
        push!(x, adv)
        push!(y, t.fix_times[end])
    end
    x, y
end

function make_featurizer(feature::Function, bins=nothing)
    hx, hy = feature(trials)
    bins = make_bins(bins, hx)
    (sim) -> begin
        try
            mxy = feature(sim)
            bin_by(bins, mxy...) .|> mean
        catch
            missing
        end
    end
end

function n_fix_hist(trials)
    n_fix = length.(trials.fix_times)
    1:10, counts(n_fix, 10) ./ length(n_fix)
end

rt_hist = let
    n = 8
    bins = make_bins(n, trials.rt)
    function rt_hist(sim)
        rt = sum.(sim.fix_times)
        x = bins.(rt) |> skipmissing |> collect |> counts
        1:n, x / length(rt)
    end
end

featurizers = Dict(
    :value_choice => make_featurizer(value_choice),
    :fixation_bias => make_featurizer(fixation_bias),
    :value_bias => make_featurizer(value_bias),
    :fourth_rank => make_featurizer(fourth_rank, :integer),
    :first_fixation_duration => make_featurizer(first_fixation_duration),
    :last_fixation_duration => make_featurizer(last_fixation_duration),
    :difference_time => make_featurizer(difference_time),
    :difference_nfix => make_featurizer(difference_nfix),
    :fixation_times => make_featurizer(fixation_times, :integer),
    :last_fix_bias => make_featurizer(last_fix_bias),
    :gaze_cascade => make_featurizer(gaze_cascade, :integer),
    :fixate_on_best => make_featurizer(fixate_on_best, Binning(0:CUTOFF/7:CUTOFF)),
    :n_fix_hist => make_featurizer(n_fix_hist, :integer),
    :rt_hist => make_featurizer(rt_hist, :integer)
)

compute_features(sim) = Dict(name => f(sim) for (name, f) in featurizers)
