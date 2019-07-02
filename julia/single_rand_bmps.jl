include("bmps_moments_fitting.jl")
x = rand(3)
results = Results("rand_bmps")
@time save(results, :xy, (x=x, y=loss(x; no_memo=true, verbose=true)))