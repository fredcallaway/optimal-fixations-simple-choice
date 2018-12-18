include("model.jl")

m = MetaMDP(obs_sigma=10, switch_cost=3)
pol = MetaGreedy(m)

function sample_trial()
    fix = Int[]
    choice = rollout(m, pol, callback=(b,c)->push!(fix, c)).choice
    (choice=choice, fixations=fix[1:end-1])  # last c is 0 (termination action)
end

trial = sample_trial()