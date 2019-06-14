include("normal_metabandits.jl")

n_obs(m, b) = round(Int, sum(((d.σ ^ -2) - 1) * m.σ_obs^2 for d in b.value))

function symmetry_breaking_hash(b::Belief)
    key = UInt64(0)
    for i in 1:length(b)
        key += (hash(b[i]) << 3(i == b.focused))
    end
    key
end


function unique_beliefs(m, depth, h)
    seen = Set{UInt64}()
    frontier = Dict{UInt64, Belief}()
    b = Belief(m)
    frontier[h(b)] = b
    while !isempty(frontier)
        b = pop!(frontier)[2]
        is_terminal(b) && continue
        push!(seen, h(b))
        n_obs(m, b) > depth && continue
        for c in actions(m)
            for (p, b1, r) in results(m, b, c)
                frontier[h(b1)] = b1
            end
        end
    end
    seen
end

# %% ====================  ====================
function explore(m, depth, h, min_voi)
    # seen = Set{UInt64}()
    # seen = Set{Belief{3}}()
    seen = Belief{3}[]
    frontier = Dict{UInt64, Belief}()
    cs = reverse(actions(m)[2:end])
    b = Belief(m)
    frontier[h(b)] = b
    while !isempty(frontier)
        b = pop!(frontier)[2]
        is_terminal(b) && continue
        push!(seen, b)
        n_obs(m, b) >= depth && continue
        for c in cs
            voi_action(b, c) < min_voi && continue
            for (p, b1, r) in results(m, b, c)
                frontier[h(b1)] = b1
            end
        end
    end
    seen
end


# %% ====================  ====================
using Printf
beliefs = explore(m, 2, symmetry_breaking_hash, 0.001)
for b in beliefs
    for i in 1:3
        @printf "%+2.1f " b.value[i].μ
    end
    println()
end
# %% ====================  ====================
n_belief(q, d, mv=0, μd=10) = length(explore(MetaMDP{3}(quantization=q, μ_digits=μd), d, symmetry_breaking_hash, mv))
q = 5
println("min_voi = 0.001;  μ_digts = 1")
for d in 1:20
    @time println("d = $d;  ", n_belief(q, d, .001, 1), " unique states");
end
# %% ====================  ====================
result(b, c, i) = results(m, b, c)[i][2]
result(b, cis) = reduce((b, ci) -> result(b, ci...), cis; init=b)
result(b, [(1,1), (1,2)])
result(b, [(1,2), (1,1)])
# %% ====================  ====================
m = MetaMDP{3}(σ_obs=2, μ_digits=1, quantization=4)
for x in results(m, Belief(m), 1)
    println(round(x[1]; digits=3), " ", x[2])
end

# %% ====================  ====================


update()


# %% ====================  ====================
m = MetaMDP{3}(quantization=4, μ_digits=1)
b = Belief(m)
c = 1
b1 = results(m, b, c)[1][2]

sad(b, c) = results(m, b, c)[1][2]

sad(b, cs::Vector{Int}) = reduce((b, c) -> sad(b, c), cs; init=b)

sad(b, [2, 1, 1]) == sad(b, [1, 2, 1])
h = symmetry_breaking_hash
h(sad(b, [2,3,3])) == h(sad(b, [2,1,1]))

# %% ====================  ====================
explore(m, 1, symmetry_breaking_hash, 1e-10)

# %% ====================  ====================

# q, d = 5, 5  with SVector
# 10.171716 seconds (44.22 M allocations: 2.001 GiB, 6.86% gc time)
# b = Belief(MetaMDP())

# b[1] = Normal(1, 0.5)
#
# heatmap(1:5, 1:5, n_belief)
#
# N = [n_belief(q, d) for q=1:5,  d=1:5]
#
# f = [n_belief(q, d) for q=1:5,  d=1:2]
# plot(N', yaxis=:log)
# plot(log.(N'))
#
#
# x = 2 .^ (1:10)
#
#
# heatmap(log.(N) ./ log(10))
# log.(N) ./ log(10)
#
# using Plots
# plot([1,2])
# # for b in unique_beliefs(m, 0)
# #     println(b)
# # end
