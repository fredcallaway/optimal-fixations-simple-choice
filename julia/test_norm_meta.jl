include("normal_metabandits.jl")

n_obs(m, b) = sum(((d.σ ^ -2) - 1) * m.σ_obs^2 for d in b.value)

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

function explore(m, depth, h, min_voi)
    seen = Set{UInt64}()
    frontier = Dict{UInt64, Belief}()
    b = Belief(m)
    frontier[h(b)] = b
    while !isempty(frontier)
        b = pop!(frontier)[2]
        is_terminal(b) && continue
        push!(seen, h(b))
        n_obs(m, b) > depth && continue
        for c in actions(m)[2:end]
            voi_action(b, c) < min_voi && continue
            for (p, b1, r) in results(m, b, c)
                frontier[h(b1)] = b1
            end
        end
    end
    seen
end
n_belief(q, d, mv=0, μd=10) = length(explore(MetaMDP{3}(quantization=q, μ_digits=μd), d, symmetry_breaking_hash, mv))

# %% ====================  ====================
q = 5
println("min_voi = 0.001;  μ_digts = 1")
for d in 1:20
    @time println("d = $d;  ", n_belief(q, d, .001, 1), " unique states");
end
# %% ====================  ====================
m = MetaMDP{3}(quantization=4, μ_digits=1)
b = Belief(m)
c = 1
results(m, b, c)


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
