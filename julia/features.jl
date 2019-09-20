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
        low, high = quantile(hx, [0.025, 0.975])
        bin_size = (high - low) / bins
        bins = Binning(low:bin_size:high)
        # bins = Binning(quantile(hx, 0:1/bins:1))
    end
    return bins
end

function total_fix_times(t)::Vector{Float64}
    x = zeros(3)
    for (fi, ti) in zip(t.fixations, t.fix_times)
        x[fi] += ti
    end
    return x
end

function relative_value(t)
    t.value .- mean(t.value)
end

function fixation_times(trials, n)
    x = Tuple{Int, Float64}[]
    for t in trials
        f = t.fix_times
        length(f) != n && continue
        for i in 1:n
            push!(x, (i, f[i]))
        end
    end
    invert(x)
end

function binned_fixation_times(trials)
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
        # length(f) > 1 && push!(x, (6, f[end-1]))
    end
    invert(x)
end

function chosen_fix_time(trials)
    mapmany(trials) do t
        map(t.fixations, t.fix_times) do f, ft
            (f == t.choice, ft)
        end
    end |> invert
end

function value_duration(trials)
    mapmany(trials) do t
        # rv = t.value .- mean(t.value)
        map(t.fixations, t.fix_times) do f, ft
            (t.value[f], ft)
        end
    end |> invert
end

function value_duration_first(trials)
    map(trials) do t
        # rv = t.value .- mean(t.value)
        f, ft = t.fixations[1], t.fix_times[1]
        (t.value[f], ft)
    end |> invert
end




function fixation_bias(trials)
    mapmany(trials) do t
        ft = total_fix_times(t)
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
        tft = total_fix_times(t)
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


function fixate_on_best(trials; sample_time=10, cutoff=2000, n_bin=5)
    n_sample = Int(cutoff / sample_time)
    spb = Int(n_sample/n_bin)
    x = Int[]
    y = Float64[]
    for t in trials
        sum(t.fix_times) < cutoff && continue
        fix = discretize_fixations(t; sample_time=sample_time)
        fix_best = fix[1:n_sample] .== argmax(t.value)
        push!(x, (1:n_bin)...)
        push!(y, mean.(Iterators.partition(fix_best, spb))...)
    end
    x, y
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
        tft = total_fix_times(t)
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

function value_bias_split(trials; chosen=false)
    x = Float64[]
    y = Float64[]
    for t in trials
        rv = relative_value(t)
        tft = total_fix_times(t)
        pft = tft ./ (sum(tft) + eps())
        for i in 1:3
            if chosen == (i == t.choice)
                push!(x, rv[i])
                push!(y, pft[i])
            end
        end
    end
    x, y
end

nfix(t) = length(t.fixations)
selectors = Dict(
    "all" => (t, i) -> true,
    "first" => (t, i) -> i == 1,
    "nonfinal" => (t, i) -> i != nfix(t),
    "final" => (t, i) -> i == nfix(t),
    "middle" => (t, i) -> i > 2 && i != nfix(t),
)

function value_duration_alt(trials; relative=false, selector=(t, i)->true)
    if selector isa String
        selector = selectors[selector]
    end
    x, y = Float64[], Float64[]
    for t in trials
        for i in eachindex(t.fixations)
            if selector(t, i)
                v = relative ? relative_value(t) : t.value
                push!(x, v[t.fixations[i]])
                push!(y, t.fix_times[i])
            end
        end
    end
    x, y
end

function fixate_on_worst(trials; sample_time=10, cutoff=2000, n_bin=5)
    n_sample = Int(cutoff / sample_time)
    spb = Int(n_sample/n_bin)
    x = Int[]
    y = Float64[]
    for t in trials
        sum(t.fix_times) < cutoff && continue
        fix = discretize_fixations(t; sample_time=sample_time)
        fix_worst = fix[1:n_sample] .== argmin(t.value)
        push!(x, (1:n_bin)...)
        push!(y, mean.(Iterators.partition(fix_worst, spb))...)
    end
    x, y
end

function fixation_bias_corrected(trials)
    v, c = value_choice(trials)
    bins = bins = Binning(v, 10)

    p_choice = bin_by(bins, v, c) .|> mean
    x = Float64[]; y = Float64[]
    for t in trials
        ft = total_fix_times(t)
        b = bins.(relative_value(t))
        p_val = p_choice[b]
        corrected = (t.choice .== 1:3) - p_val
        ft .-= mean(ft)
        # ft ./= sum(ft)
        for i in 1:3
            push!(x, ft[i])
            push!(y, corrected[i])
        end
    end
    x, y
end

function full_fixation_times(trials)
    x = Int[]
    y = Float64[]
    for t in trials
        for (i, f) in enumerate(t.fix_times)
            i > 10 && break
            push!(x, i)
            push!(y, f)
        end
    end
    x, y
end

function refixate_uncertain(trials)
    options = Set([1,2,3])
    x = Float64[]
    for t in trials
        cft = zeros(3)
        total = 0
        for i in eachindex(t.fixations)
            fix = t.fixations[i]
            fix_time = t.fix_times[i]
            if i > 2
                prev = t.fixations[i-1]
                alt = pop!(setdiff(options, [prev, fix]))
                push!(x, cft[fix] - cft[alt])
            end
            cft[fix] += fix_time
            total += fix_time
        end
    end
    return x
end
