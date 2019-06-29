using DataStructures: OrderedDict

struct Box
    dims::OrderedDict
end

Box(dims...) = Box(OrderedDict(dims))
Base.length(b::Box) = length(b.dims)
Base.getindex(box::Box, k) = box.dims[k]

function Base.display(box::Box)
    println("Box")
    for p in pairs(box.dims)
        println("  ", p)
    end
end

linscale(x, low, high) = low + x * (high-low)
logscale(x, low, high) = exp(log(low) + x * (log(high) - log(low)))

unlinscale(x, low, high) = (x - low) / (high-low)
unlogscale(x, low, high) = (log(x) - log(low)) / (log(high) - log(low))

function rescale(d, x)
    scale = :log in d ? logscale : linscale
    scale(x, d[1], d[2])
end

function unscale(d, x)
    scale = :log in d ? unlogscale : unlinscale
    scale(x, d[1], d[2])
end

n_free(b::Box) = sum(length(d) > 1 for d in values(b.dims))

function (box::Box)(x)
    @assert length(x) == n_free(box)

    xs = Iterators.Stateful(x)
    map(collect(box.dims)) do (name, d)
        if length(d) > 1
            name => rescale(d, popfirst!(xs))
        else
            name => d
        end
    end |> OrderedDict
end
