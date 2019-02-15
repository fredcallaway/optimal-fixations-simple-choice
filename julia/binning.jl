struct Binning{T}
    limits::Vector{T}
end
Binning(b::AbstractVector) = Binning(collect(b))
Base.length(bins::Binning) = length(bins.limits) - 1

Binning(xs, n) = begin
    lims = quantile(xs, range(0, 1, length=n+1))
    lims[end] *= 1.00001
    Binning(lims)
end
mids(bins::Binning) = [mean(bins.limits[i:i+1]) for i in 1:length(bins.limits)-1]

(bins::Binning)(x) = begin
    idx = findfirst(x .< bins.limits)
    if idx == nothing || idx == 1
        return missing
    else
        return idx - 1
    end
end
bin(xs, n) = Binning(xs, n).(xs)

function bin_means(x, y; n=5)
    bins = bin(x, n)
    grp = group(x->x[1], x->x[2], zip(bins, y)) |> sort
    yy = grp |> values |> collect .|> mean
    xx = grp |> keys |> collect
    (xx, yy)
end

function bin_by(bins::Binning, x, y) where T
    grp = group(i->bins(x[i]), i->y[i], 1:length(x))
    typeof(y)[get(grp, i, []) for i in 1:length(bins)]
end
