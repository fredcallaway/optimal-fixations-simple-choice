using DataStructures: OrderedDict

struct Box
    dims::OrderedDict
end

Box(dims...) = Box(OrderedDict(dims))
Base.length(b::Box) = length(b.dims)
function Base.display(box::Box)
    println("Box")
    for p in pairs(box.dims)
        println("  ", p)
    end
end

linscale(x, low, high) = low + x * (high-low)
logscale(x, low, high) = exp(log(low) + x * (log(high) - log(low)))

function rescale(d, x)
    scale = :log in d ? logscale : linscale
    scale(x, d[1], d[2])
end


function (box::Box)(x)
    @assert length(x) == length(box)
    map(box.dims, x) do d, xi
        d[1] => rescale(d[2], xi)
    end |> OrderedDict
end
