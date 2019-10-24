total_fix_time(t) = sum(t.fix_times)
rank_chosen(t) = sortperm(sortperm(t.value; rev=true))[t.choice]
n_fix(t) = length(t.fixations)

function chosen_fix_proportion(t)
    isempty(t.fixations) && return 0.
    tft = total_fix_times(t)
    tft ./= sum(tft)
    tft[t.choice]
end

function top_fix_proportion(t)
    isempty(t.fixations) && return 0.
    tft = total_fix_times(t)
    tft ./= sum(tft)
    tft[argmax(t.value)]
end

struct Metric{F}
    f::F
    bins::Binning
end

function Metric(f::Function, n::Int)
    bins = Binning(f.(rank_trials), n)
    bins.limits[1] = -Inf; bins.limits[end] = Inf
    Metric(f, bins)
end
(m::Metric)(t) = t |> m.f |> m.bins

# m = Metric(total_fix_time, 10)
# counts(m.(trials))  # something's wrong?


function final_fix_times(t)::Vector{Float64}
    x = zeros(3)
    for (fi, ti) in zip(t.fixations, t.fix_times)
        x[fi] += ti
    end
    return x
end

function fix_time_var(t)
    isempty(t.fixations) && return 0.
    fft = final_fix_times(t)
    fft ./= sum(fft)
    var(fft)
end

function propfix(t)
    prop = total_fix_times(t)
    prop ./= sum(prop)
end
