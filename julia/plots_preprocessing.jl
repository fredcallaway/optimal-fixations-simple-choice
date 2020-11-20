include("plots_features.jl")
include("human.jl")
using Serialization
using StatsBase

N_BOOT = 10000
CI = 0.95
USE_SEM = false
function ci_err(y)
    isempty(y) && return (NaN, NaN)
    length(y) == 1 && return (0., 0.)
    std(y) ≈ 0 && return (0., 0.)
    USE_SEM && return (2sem(y), 2sem(y))
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
    @show feature
    key = (feature=feature, feature_kws...)
    if (length(trials[1].value) == 2) && three_only
        return key => missing
    end
    key => compute_feature(feature, trials; bin_spec=bin_spec, feature_kws...)
end

@memoize function get_bins(feature, n_item, bin_spec, feature_kws)
    trials = load_dataset(n_item, :test)
    hx, hy = feature(trials; feature_kws...)
    bins = make_bins(bin_spec, hx)
end

function compute_feature(feature::Function, trials; bin_spec=:integer, feature_kws...)
    n_item = length(trials[1].value)
    bins = get_bins(feature, n_item, bin_spec, feature_kws)
    x, y = feature(trials; feature_kws...)
    vals = bin_by(bins, x, y)
    err = ci_err.(vals)
    (x=mids(bins), y=mean.(vals), err=err, n=length.(vals))
end

function compute_plot_features(trials)
    f = trials isa Table ? compute_feature_one : compute_feature_both
    Dict(
        f(first_fixation_duration_corrected, trials; bin_spec=7),
        f(fixation_bias_corrected, trials; bin_spec=7),
        f(value_choice, trials),
        f(difference_time, trials),
        f(meanvalue_time, trials),
        f(nfix_hist, trials),
        f(difference_nfix, trials),
        f(binned_fixation_times, trials),
        f(fixate_by_uncertain, trials; bin_spec=7, three_only=true),
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

function compute_plot_features_individual(trials)
    f = trials isa Table ? compute_feature_one : compute_feature_both
    Dict(
        f(value_bias, trials; bin_spec=Binning(-6:4:6)),
        f(fix4_value, trials; bin_spec=Binning(-6:4:6), three_only=true),
        f(value_duration, trials; bin_spec=Binning(0:3:9), fix_select=firstfix),
        f(value_choice, trials; bin_spec=Binning(-6:4:6)),
        f(binned_fixation_times, trials; bin_spec=:integer),
        f(nfix_hist, trials; bin_spec=:integer),
        f(last_fix_bias, trials; bin_spec=Binning(-6:4:6)),
        f(fixation_bias, trials; bin_spec=Binning(-1500:1500:1500)),
    )
end


