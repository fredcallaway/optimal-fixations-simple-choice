include("features.jl")

function mpe(y, yhat)
    @assert size(y) == size(yhat)
    mean(abs.(y .- yhat) ./ y)
end

function visual_error(y, yhat)
    scale = maximum(y) - minimum(y)
    mean(abs.(y .- yhat) ./ scale)
end

function make_loss(feature::Function, bins=nothing)
    hx, hy = feature(trials)
    bins = make_bins(bins, hx)
    h = bin_by(bins, hx, hy) .|> mean
    (sim) -> begin
        mxy = feature(sim)
        ismissing(mxy) && return Inf
        m = bin_by(bins, mxy...) .|> mean
        err = visual_error(h, m)
        isnan(err) ? Inf : err
    end
end

# loss_funcs = [
#     make_loss(fixation_bias),
#     make_loss(difference_nfix),
#     make_loss(value_choice, :integer),
#     # make_loss(first_fixation_bias),
#     make_loss(last_fix_bias),
#     make_loss(value_bias),
#     make_loss(gaze_cascade, :integer),
#     make_loss(fixation_value, Binning(0:3970/20:3970)),
# ]
#
# function loss(sim)
#     sum(l(sim) for l in loss_funcs)
# end
