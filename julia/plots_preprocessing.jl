include("plots_features.jl")
include("human.jl")
using Serialization
using StatsBase

N_BOOT = 10000
CI = 0.95
function ci_err(y)
    length(y) == 1 && return (0., 0.)
    isempty(y) && return (NaN, NaN)
    # return (2sem(y), 2sem(y))
    method = length(y) <= 10 ? ExactSampling() : BasicSampling(N_BOOT)
    bs = bootstrap(mean, y, method)
    c = confint(bs, BCaConfInt(CI))[1]
    abs.(c[2:3] .- c[1])
end

function compute_feature_both(feature::Function, both_trials; bin_spec=:integer, three_only=false, feature_kws...)
    idx = three_only ? [2] : [1, 2]
    key = (feature=feature, feature_kws...)
    key => map(both_trials[idx]) do trials
        n_item = length(trials[1].value)
        n_item => compute_feature(feature, trials; bin_spec=bin_spec, feature_kws...)
    end |> Dict
end

function compute_feature_one(feature, trials; bin_spec=:integer, three_only=false, feature_kws...)
    key = (feature=feature, feature_kws...)
    if (length(trials[1].value) == 2) && three_only
        return key => missing
    end
    key => compute_feature(feature, trials; bin_spec=bin_spec, feature_kws...)

end

function compute_feature(feature::Function, trials; bin_spec=:integer, feature_kws...)
    mx, my = feature(trials; feature_kws...)
    @assert false
    hx, hy = feature(trials; feature_kws...)
    bins = make_bins(bin_spec, hx)
    vals = bin_by(bins, mx, my)
    err = ci_err.(vals)
    (x=mids(bins), y=mean.(vals), err=err, n=length.(vals))
end

function compute_plot_features(trials)
    f = trials isa Table ? compute_feature_one : compute_feature_both
        Dict(
            f(first_fixation_duration_corrected, trials; bin_spec=7),
            f(fixation_bias_corrected, trials; bin_spec=7),
            f(value_choice, trials; bin_spec=Binning(-4.5:1:4.5)),
            f(difference_time, trials),
            f(nfix_hist, trials),
            f(difference_nfix, trials),
            f(binned_fixation_times, trials),
            f(fixate_by_uncertain, trials; bin_spec=Binning(-50:100:850), three_only=true),
            f(value_bias, trials),
            f(value_duration, trials; fix_select=firstfix),
            f(fixate_on_worst, trials; cutoff=2000, n_bin=20),
            f(fix4_value, trials; three_only=true),
            f(fix3_value, trials; three_only=true),
            f(last_fix_bias, trials),
            f(fixation_bias, trials; bin_spec=7),
            f(first_fixation_duration, trials; bin_spec=7),
        )
    end