using Printf
using Parameters

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

Base.dropdims(idx::Int...) = X -> dropdims(X, dims=idx)
Base.reshape(idx::Union{Int,Colon}...) = x -> reshape(x, idx...)

# type2dict(x::T) where T = Dict(fn=>getfield(x, fn) for fn in fieldnames(T))

function background(f, name; save=false)
    @async begin
        try
            x, t = @timed f()
            println(name, " finished in ", round(t), " seconds")
            mkpath("background_tasks")
            serialize("background_tasks/$name", x)
            return x
        catch e
            println("ERROR: $name failed")
            rethrow(e)
        end
    end
end
