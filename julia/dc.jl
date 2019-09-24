include("utils.jl")
include("voi.jl")

using StatsBase

function voc_dc(m::MetaMDP, b::Belief, c::Computation)
    c == ⊥ && return 0.
    voc_n(n) = voi_n(b, c, n) - (cost(m, b, c) + (n-1) * m.sample_cost)
    # int_line_search(1, voc_n)[2]
    maximum(voc_n.(1:100))
end

struct DirectedCognition <: Policy
    m::MetaMDP
end

action(pol::DirectedCognition, b::Belief) = begin
    voc = [voc_dc(pol.m, b, c) for c in 1:pol.m.n_arm]
    v, c = findmax(noisy(voc))
    v <= 0 ? ⊥ : c
end
(pol::DirectedCognition)(b::Belief) = action(pol, b)


struct SoftDC <: Policy
    m::MetaMDP
    α::Float64
end
voc(pol::SoftDC, b::Belief) = [voc_dc(pol.m, b, c) for c in 0:pol.m.n_arm]
action_probs(pol::SoftDC, b::Belief) = softmax(pol.α * voc(pol, b))

(pol::SoftDC)(b::Belief) = begin
    sample(0:pol.m.n_arm, Weights(action_probs(pol, b)))
end
