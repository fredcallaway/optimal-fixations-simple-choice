Base.map(f, d::AbstractDict) = [f(k, v) for (k, v) in d]
mapvalues(f, d::AbstractDict) = Dict(k=> f(v) for (k,v) in d)
juxt(fs...) = x -> Tuple(f(x) for f in fs)
repeatedly(f, n) = [f() for i in 1:n]
function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end
