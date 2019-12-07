using Printf
function describe_vec(x::Vector)
    @printf("%.3f ± %.3f  [%.3f, %.3f]\n", juxt(mean, std, minimum, maximum)(x)...)
end

Base.map(f, d::AbstractDict) = [f(k, v) for (k, v) in d]
valmap(f, d::AbstractDict) = Dict(k => f(v) for (k, v) in d)
valmap(f) = d->valmap(f, d)
keymap(f, d::AbstractDict) = Dict(f(k) => v for (k, v) in d)
juxt(fs...) = x -> Tuple(f(x) for f in fs)
repeatedly(f, n) = [f() for i in 1:n]

dictkeys(d::AbstractDict) = (collect(keys(d))...,)
dictvalues(d::AbstractDict) = (collect(values(d))...,)
namedtuple(d::AbstractDict) = NamedTuple{dictkeys(d)}(dictvalues(d))

function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

# import Serialization: serialize
# function serialize(s::String, x)
#     open(s, "w") do f
#         Serialization.serialize(f, x)
#     end
# end