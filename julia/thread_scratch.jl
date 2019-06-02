include("normal_metabandits.jl")
m = MetaMDP()
b = Belief(m)
@time update(m, b, 1)

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

using Base.Threads

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
        @threads for c in actions(m)[2:end]
            voi_action(b, c) < min_voi && continue
            for (p, b1, r) in results(m, b, c)
                frontier[h(b1)] = b1
            end
        end
    end
    seen
end
n_belief(q, d, mv=0, μd=10) = length(explore(MetaMDP{3}(quantization=q, μ_digits=μd), d, symmetry_breaking_hash, mv))
@time n_belief(3, 7)


@time length(unique_beliefs(m, 4, symmetry_breaking_hash))
