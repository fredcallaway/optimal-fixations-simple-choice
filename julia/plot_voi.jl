using Distributed
using StatsPlots

nprocs() == 1 && addprocs()
pyplot(label="", grid=:none)
@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
end

m = MetaMDP(sample_cost=0., σ_obs=2)
function term_reward_curve(N)
    pol = BMPSPolicy(m, [0., 1., 0., 0.])
    trc = @distributed (+) for i in 1:Int(N/100)
        x = zeros(101)
        for i in 1:100
            roll = rollout(pol; max_steps=101) do b, c
                i = 1+Int((sum(b.λ) - 3.) / (m.σ_obs^-2))
                x[i] += term_reward(b)
            end
            @assert roll.steps == 101
        end
        x
    end
    trc ./ N
end


# %% ====================  ====================
trc = term_reward_curve(Int(1e6))
# %% ====================  ====================
Plots.scalefontsizes(); Plots.scalefontsizes(2)
function line!(v, lab, c=:black)
    hline!([v], c=c, line=(:dash, 3))
    annotate!(50, v, text(lab, 24, :center, :bottom), c=c)
end

# Plots.scalefontsizes(1.5)
plot(0:100, trc, c=:black, ylim=(0, 1),size=(900,600),
    lw=2, xlabel="Number of Computations", ylabel="Value of Chosen Item",
    xticks=([1,25,50,75,100], [1,25,50,75,100]))
line!(vpi(Belief(m)), L"\mathrm{VOI}_\mathrm{full}")
line!(voi1(Belief(m), 1), L"\mathrm{VOI}_\mathrm{myopic}")
plot!([1, 1], [0, trc[2]], line=(:solid, :black, 1))


# line!(voi_action(Belief(m), 1), L"\mathrm{VOI}_\mathrm{item}")
savefig("figs/voi-vpi.pdf")
