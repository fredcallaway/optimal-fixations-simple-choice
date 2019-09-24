using Distributed
using SharedArrays
using Printf
using Sobol
using OnlineStats
using Statistics

function max_cost(m::MetaMDP)
    θ = [1., 0, 0, 1]
    b = Belief(m)
    # s = State(m)
    # b = Belief(s)
    function computes()
        pol = BMPSPolicy(m, θ)
        all(pol(b) != ⊥ for i in 1:30)
    end

    while computes()
        θ[1] *= 2
    end

    while !computes()
        θ[1] /= 2
        if θ[1] < 2^-10
            error("Computation is too expensive")
        end
    end

    step_size = θ[1] / 10
    while computes()
        θ[1] += step_size
    end
    θ[1]
end


function x2theta(mc, x)
    voi_weights = diff([0; sort(collect(x[2:3])); 1])
    [x[1] * mc; voi_weights]
end

include("gp_bmps.jl")






