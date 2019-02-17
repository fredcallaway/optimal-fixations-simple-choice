include("features.jl")

function mpe(y, yhat)
    @assert size(y) == size(yhat)
    mean(abs.(y .- yhat) ./ y)
end

function make_loss(feature::Function, bins=nothing)
    hx, hy = feature(trials)
    bins = make_bins(bins, hx)
    h = bin_by(bins, hx, hy) .|> mean
    (sim) -> begin
         m = bin_by(bins, feature(sim)...) .|> mean
         err = mpe(h, m)
         isnan(err) ? Inf : err
    end
end


losses = [
    make_loss(fixation_bias),
    make_loss(difference_time),
    make_loss(value_choice, :integer),
    make_loss(first_fixation_bias),
    make_loss(last_fix_bias),
    make_loss(value_bias),
]

function loss(sim)
    sum(l(sim) for l in losses)
end
