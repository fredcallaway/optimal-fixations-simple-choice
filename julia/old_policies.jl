struct MetaGreedy
    m::MetaMDP
end

(pol::MetaGreedy)(b::Belief) = begin
    voc1 = [voi1(b, c) - cost(pol.m, b, c) for c in 1:pol.m.n_arm]
    v, c = findmax(noisy(voc1))
    v <= 0 ? ⊥ : c
end

struct Noisy{T}
    ε::Float64
    π::T
    m::MetaMDP
end
Noisy(ε, π) = Noisy(ε, π, π.m)

(π::Noisy)(b::Belief) = begin
    rand() < π.ε ? rand(1:length(b.mu)) : π.π(b)
end

# %% ====================  ====================
mutable struct FixedPolicy
    m::MetaMDP
    plan::Vector{Int}
    state::Int
end
FixedPolicy(m, plan) = FixedPolicy(m, plan, 1)

(pol::FixedPolicy)(b::Belief) = begin
    pol.state > length(pol.plan) && return ⊥
    c = pol.plan[pol.state]
    pol.state += 1
    c
end

function reset!(pol::FixedPolicy)
    pol.state = 1
end

