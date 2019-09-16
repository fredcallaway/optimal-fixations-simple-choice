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


function apply(box::Box, x::Vector{Float64})
    xs = Iterators.Stateful(x)
    map(collect(box.dims)) do (name, dim)
        if length(dim) > 1
            name => rescale(dim, popfirst!(xs))
        else
            name => dim
        end
    end |> OrderedDict
end

function apply(box::Box, d::AbstractDict)
    x = Float64[]
    for (name, dim) in box.dims
        if length(dim) > 1
            push!(x, unscale(dim, d[name]))
        end
    end
    return x
end

(box::Box)(x) = apply(box, x)
