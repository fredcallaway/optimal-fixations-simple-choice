include("features.jl")

const CUTOFF = 2000

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
        try
            mxy = feature(sim)
            ismissing(mxy) && return Inf
            m = bin_by(bins, mxy...) .|> mean
            err = visual_error(h, m)
            isnan(err) ? Inf : err
        catch
            Inf
        end
    end
end

loss_funcs = [
    make_loss(value_choice),
    make_loss(fixation_bias),
    make_loss(value_bias),
    make_loss(fourth_rank, :integer),
    make_loss(first_fixation_duration),
    make_loss(last_fixation_duration),
    make_loss(difference_time),
    make_loss(fixation_times, :integer),
    make_loss(last_fix_bias),
    make_loss(gaze_cascade, :integer),
    make_loss(fixate_on_best, Binning(0:CUTOFF/7:CUTOFF)),
    # make_loss(difference_nfix),
    # make_loss(old_value_choice, :integer),
    # make_loss(fixation_value, Binning(0:3970/20:3970)),
]
loss(sim) = sum(ℓ(sim) for ℓ in _loss_funcs)
breakdown_loss(sim) = [ℓ(sim) for ℓ in _loss_funcs]
