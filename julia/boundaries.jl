include("model.jl")
using Plots
using LaTeXStrings

rollout(m, pol)
# %% ====================  ====================
pol = Policy(m, [0., 0.9, 0., 0.1])
function n_step(σ, c)
    m = MetaMDP(n_arm=2, obs_sigma=σ, sample_cost=c)
    pol = MetaGreedy(m)
    mean(rollout(m, pol).steps for i in 1:100)
    # mean(rollout(m, pol).belief.lam[1] for i in 1:100)
end

n_step(3, 1e-11)

# %% ====================  ====================

"Lowest x s.t low < x high and f(x) is true"
function bisection_search(f, low, high; tol=0.01)
    println(low, high)
    @assert f(high)
    @assert !f(low)
    while high - low > tol
        mid = (low + high) / 2
        if f(mid)
            high = mid
        else
            low = mid
        end
    end
    return high
end

# %% ==================== Collapsing boundaries ====================
λ = exp.(0:0.05:4)
d = map(λ) do λ
    bisection_search(0., 3.) do x
        pol(Belief([0., x], [λ, λ], ones(2), 1)) == 0
    end
end
plot(λ, d,
     # xscale=:log,
     ylims=(0,2)
 )
xlabel!(L"\lambda")
ylabel!("Difference")
