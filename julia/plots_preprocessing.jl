include("plots_features.jl")
include("human.jl")
using Serialization
using StatsBase

N_BOOT = 10000
CI = 0.95
function ci_err(y)
    length(y) == 1 && return (0., 0.)
    isempty(y) && return (NaN, NaN)
    # return (sem(y), sem(y))
    method = length(y) <= 10 ? ExactSampling() : BasicSampling(N_BOOT)
    bs = bootstrap(mean, y, method)
    c = confint(bs, BCaConfInt(CI))[1]
    abs.(c[2:3] .- c[1])
end

both_trials = map(["two", "three"]) do num
    get_fold(load_dataset(num), "odd", :test)
end

function precompute(feature::Function, sims; bin_spec=:integer, three_only=false, feature_kws...)
    idx = three_only ? [2] : [1, 2]
    key = (feature=feature, feature_kws...)
    key => map(idx) do i
        # try
            mx, my = feature(sims[i]; feature_kws...)
            hx, hy = feature(both_trials[i]; feature_kws...)
            bins = make_bins(bin_spec, hx)
            vals = bin_by(bins, mx, my)
            err = ci_err.(vals)
            n_item = length(both_trials[i][1].value)
            n_item => (x=mids(bins), y=mean.(vals), err=err, n=length.(vals))
        # catch e
        #     println("ERROR on $feature, $(i+1) items, $e")
        #     serialize("tmp/bad_pre", sims[i])
        #     return
        # end
    end |> Dict
end
