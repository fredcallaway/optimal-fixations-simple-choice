include("binning.jl")
using StatsBase
const CUTOFF = 2000

# %% ==================== Utilities ====================

function make_bins(bins, hx)
    if bins == :integer
        low, high = quantile(hx, [0.025, 0.975])
        lo = floor(low - 0.5) + 0.5
        hi = ceil(high + 0.5) - 0.5
        return Binning(lo:1:hi)
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

final(t,i) = i == length(t.fixations)
nonfinal(t, i) = i != length(t.fixations)
allfix(t, i) = true
firstfix(t, i) = i == 1
n_item_(t) = length(t.value)

function difficulty(v)
    v = sort(v; rev=true)
    v[1] - mean(v[2:end])
end

function relative_left(x)
    x[1] - mean(x[2:end])
end

relative(x) = [x[i] - mean(x[setdiff(1:length(x), i)]) for i in eachindex(x)]

function total_fix_times(t; fix_select=allfix)::Vector{Float64}
    x = zeros(n_item_(t))
    for i in eachindex(t.fixations)
        fix_select(t, i) || continue
        fi = t.fixations[i]; ti = t.fix_times[i]
        x[fi] += ti
    end
    return x
end

# %% ==================== Features ====================


function value_choice(trials)
    x = Float64[]; y = Bool[];
    for t in trials
        push!(x, relative_left(t.value))
        push!(y, t.choice == 1)
    end
    x, y
end

function difference_time(trials)
    difficulty.(trials.value), sum.(trials.fix_times)
end

function nfix_hist(trials)
    n_fix = length.(trials.fix_times)
    1:10, counts(n_fix, 10) ./ length(n_fix)
end

function difference_nfix(trials)
    difficulty.(trials.value), length.(trials.fixations)
end

function fixate_on_(trials, which; sample_time=10, cutoff=2000, n_bin=5, nonfinal=false)
    n_sample = Int(cutoff / sample_time)
    spb = Int(n_sample/n_bin)
    x = Int[]
    y = Float64[]
    for t in trials
        unique_values(t) || continue
        # if nonfinal
        #     sum(t.fix_times[1:end-1]) < cutoff && continue
        # else
        #     sum(t.fix_times) < cutoff && continue
        # end
        fix = discretize_fixations(t; sample_time=sample_time)
        n = min(length(fix), n_sample)
        n -= n % spb  # don't include partial bins

        fix_best = fix[1:n] .== which(t.value)
        bmeans = mean.(Iterators.partition(fix_best, spb))

        push!(x, eachindex(bmeans)...)
        push!(y, bmeans...)
    end
    x, y
end


fixate_on_best(trials; kws...) = fixate_on_(trials, argmax; kws...)
fixate_on_worst(trials; kws...) = fixate_on_(trials, argmin; kws...)

function value_bias(trials; fix_select=allfix)
    x = Float64[]; y = Float64[]
    for t in trials
        push!(x, relative_left(t.value))
        tft = total_fix_times(t; fix_select=fix_select)
        tft ./= (sum(tft) + eps())
        push!(y, tft[1])
    end
    x, y
end

# function refixate_uncertain(trials; compare_to_prev=true)
#     n = n_item_(trials[1])
#     options = Set(1:n)
#     x = Float64[]
#     for t in trials
#         cft = zeros(n)
#         total = 0
#         for i in eachindex(t.fixations)
#             fix = t.fixations[i]
#             fix_time = t.fix_times[i]
#             if i > 2
#                 prev = t.fixations[i-1]
#                 if n == 3 && compare_to_prev
#                     others = [i for i in options if i != fix]
#                     push!(x, cft[fix] - mean(cft[others]))
#                 else
#                     alt = n == 2 ? prev : pop!(setdiff(options, [prev, fix]))
#                     push!(x, cft[fix] - cft[alt])
#                 end
#             end
#             cft[fix] += fix_time
#             total += fix_time
#         end
#     end
#     return x
# end

function refixate_uncertain(trials; refixate_only=false, ignore_current=false)
    n = n_item_(trials[1])
    @assert !(ignore_current && n == 2)
    options = Set(1:n)
    x = Float64[]
    for t in trials
        cft = zeros(n)
        total = 0
        for i in eachindex(t.fixations)
            fix = t.fixations[i]
            fix_time = t.fix_times[i]
            if i > (ignore_current ? 2 : 1)
                prev = t.fixations[i-1]
                if !(refixate_only && cft[fix] == 0)
                    if ignore_current
                        other = pop!(setdiff(options, [prev, fix]))
                        push!(x, cft[fix] - cft[other])
                    else
                        others = [i for i in options if i != fix]
                        push!(x, cft[fix] - mean(cft[others]))
                    end
                end
            end
            cft[fix] += fix_time
            total += fix_time
        end
    end
    return x
end


function fixate_by_uncertain(trials)
    n = n_item_(trials[1])
    @assert n != 2
    options = Set(1:n)
    x = Float64[]; y = Int[]
    for t in trials
        cft = zeros(n)
        total = 0
        for i in eachindex(t.fixations)
            fix = t.fixations[i]
            fix_time = t.fix_times[i]
            if i > 1
                prev = t.fixations[i-1]
                a, b = shuffle([i for i in options if i != prev])
                d = cft[a] - cft[b]
                push!(x, abs(d))
                push!(y, fix == (d >= 0 ? a : b))
            end
            cft[fix] += fix_time
            total += fix_time
        end
    end
    return x, y
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

function full_fixation_times(trials; fix_select=allfix)
    x = Int[]
    y = Float64[]
    for t in trials
        for (i, f) in enumerate(t.fix_times)
            # i > 10 && break
            fix_select(t, i) || continue
            push!(x, i)
            push!(y, f)
        end
    end
    x, y
end

function chosen_fix_time(trials; fix_select=allfix)
    x = Bool[]; y = Float64[]
    for t in trials
        for i in eachindex(t.fixations)
            fix_select(t, i) || continue
            push!(x, t.fixations[i] == t.choice)
            push!(y, t.fix_times[i])
        end
    end
    x, y
end


function value_duration(trials; fix_select=allfix)
    x = Float64[]; y = Float64[];
    for t in trials
        for i in eachindex(t.fixations)
            fix_select(t, i) || continue
            fi = t.fixations[i]; ti = t.fix_times[i]
            push!(x, t.value[fi])
            push!(y, ti)
        end
    end
    x, y
end

function last_fixation_duration(trials)
    x = Float64[]; y = Float64[]
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

function fix4_value(trials)
    x = Float64[]; y = Float64[]; n = 3
    @assert n_item_(trials[1]) == 3  # only makes sense for three-item case
    for t in trials
        if length(t.fixations) > n && sort(t.fixations[1:n]) == 1:n && unique_values(t)
            f1, f2, f3, f4 = t.fixations
            push!(x, t.value[f1] - t.value[f2])
            push!(y, f4 == f1)
        end
    end
    x, y
end

function old_fix4_uncertain(trials)
    x = Float64[]; y = Float64[]; n = 3
    for t in trials
        if length(t.fixations) > n && sort(t.fixations[1:n]) == 1:n && unique_values(t)
            f1, f2, f3, f4 = t.fixations
            push!(x, t.fix_times[1] - t.fix_times[2])
            push!(y, f4 == f1)
        end
    end
    x, y
end

function fix3_value(trials)
    x = Float64[]; y = Float64[]; n = 3
    for t in trials
        if length(t.fixations) >= n
            f1, f2, f3 = t.fixations
            push!(x, t.value[1])
            push!(y, f3 == f1)
        end
    end
    x, y
end

function fix3_uncertain(trials)
    x = Float64[]; y = Float64[]; n = 3
    for t in trials
        if length(t.fixations) >= n
            f1, f2, f3 = t.fixations
            push!(x, t.fix_times[1])
            push!(y, f3 == f1)
        end
    end
    x, y
end
#
# median_value = flatten(trials.value) |> median
# function fix3_value(trials; median_split)
#     x = Float64[]; y = Float64[]; n = n_item_(t)
#     for t in trials
#         if length(t.fixations) >= n
#             f1, f2, f3 = t.fixations
#             comp = Dict(:top => (>), :bottom => (<=))[median_split]
#             comp(t.value[f1], median_value) || continue
#             # push!(x, tft[f1] - tft[f2])
#             push!(x, t.fix_times[1])
#             push!(y, f3 == f1)
#         end
#     end
#     x, y
# end


function fixation_bias(trials; trial_select=(t)->true)
    x = Float64[]; y = Bool[];
    for t in trials
        trial_select(t) || continue
        push!(x, relative_left(total_fix_times(t)))
        push!(y, t.choice == 1)
    end
    x, y
end

function last_fix_bias(trials)
    x, y = Float64[], Bool[]
    for t in trials
        if length(t.fixations) > 0
            last = t.fixations[end]
            push!(x, relative(t.value)[last])
            push!(y, t.choice == last)
        end
    end
    x, y
end

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


function refixate_tft(trials)
    x = Float64[]; y = Float64[]; n = n_item_(trials[1])
    for t in trials
        if length(t.fixations) > n && sort(t.fixations[1:n]) == 1:n && unique_values(t)
            f1, f2, f3, f4 = t.fixations
            tft = total_fix_times(t)
            push!(x, tft[f1] - tft[f2])
            push!(y, f4 == f1)
        end
    end
    x, y
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



function fixation_bias(trials)
    x = Float64[]; y = Bool[];
    for t in trials
        push!(x, relative_left(total_fix_times(t)))
        push!(y, t.choice == 1)
    end
    x, y
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
    p = counts(x, n_item_(t)) ./ n
    std_ = @. √(p * (1 - p) / n)
    eachindex(p), p, std_
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

function value_bias_split(trials; chosen=false)
    x = Float64[]; y = Float64[]; n = n_item_(t)
    for t in trials
        rv = relative_value(t)
        tft = total_fix_times(t)
        pft = tft ./ (sum(tft) + eps())
        for i in 1:n
            if chosen == (i == t.choice)
                push!(x, rv[i])
                push!(y, pft[i])
            end
        end
    end
    x, y
end


using Optim
using Memoize
@memoize function fit_softmax(trials)
    res = optimize(0, 5) do α
        mapreduce(+, trials.value, trials.choice) do v, c
            -log(softmax(α .* v)[c])
        end
    end
    res.minimizer
end

function fixation_bias_corrected(trials)
    α = fit_softmax(trials)
    x = Float64[]; y = Float64[]; n = n_item_(trials[1])
    for t in trials
        push!(x, relative_left(total_fix_times(t)))
        correction = softmax(α .* t.value)[1]
        push!(y, (t.choice == 1) - correction)
    end
    x, y
end


function first_fixation_duration_corrected(trials)
    α = fit_softmax(trials)
    x, y = Float64[], Float64[]
    for t in trials
        if length(t.fixations) > 0
            push!(x, t.fix_times[1])
            correction = softmax(α .* t.value)[t.fixations[1]]
            push!(y, (t.choice == t.fixations[1]) - correction)
        end
    end
    x, y
end
