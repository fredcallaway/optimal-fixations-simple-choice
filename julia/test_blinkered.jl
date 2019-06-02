include("blinkered.jl")

m = MetaMDP(obs_sigma=1.1, sample_cost=.002, switch_cost=3.)

# %% ====================  ====================
function test()
    for i in 1:100000
        b = Belief(State(m))
        b.mu .= randn(3)
        b.lam .+= rand(3)
        try
            for c in 1:3
                @assert _voc_blinkered(m, b, c) ≈ voc_blinkered(m, b, c)
            end
        catch
            return b
        end
    end
end

b = test()
for c in 1:3
    @show _voc_blinkered(m, b, c)
    @show voc_blinkered(m, b, c)
end