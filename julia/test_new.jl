include("bernoulli_metabandits.jl")

# %% ====================  ====================
m = MetaMDP(sample_cost=0.1)
b = Belief(m)
display("")
println(b)
println(voi1(m, b, 1))
println(voi_action(m, b, 1))
println(vpi(m, b))
# %% ====================  ====================
m = MetaMDP(sample_cost=.01, max_obs=30)
b = Belief(m)
V = ValueFunction(m)
@time println(V(b))
# %% ====================  ====================
m = MetaMDP(sample_cost=.01, max_obs=30)
pol = OptimalPolicy(m)

function expectation(f, n)
    acc = 0.
    for i in 1:n
        acc += f()
    end
    acc / n
end

# %% ====================  ====================
function show_rollouts(m)
    pol = OptimalPolicy(m)
    for _ in 1:5
        rollout(pol) do b, c
            print(c, " ")
        end
        println()
    end
end

function cutoff_rate(pol)
    map(1:1000) do i
        rollout(pol).steps == m.max_obs + 1
    end |> mean
end


let
    m = MetaMDP(sample_cost=0.0003, switch_cost=1, max_obs=50)
    pol = OptimalPolicy(m)
    # pol.V(Belief(m))
    c = cutoff_rate(pol)
    print("Expected steps: ")
    expectation(1000) do
        rollout(pol).steps
    end |> println
    c > 0.01 && println("CUTOFF RATE: $c")
    show_rollouts(m)
end


m = MetaMDP(sample_cost=.001, switch_cost=5, max_obs=50)
result = optimize(m, n_iter=10)
id = round(Int, rand() * 1e8)
open("bernoulli/$id", "w+") do f
    serialize(f, (m=m, result=result, time_stamp=now()))
end


# %% ====================  ====================
possible = actions(pol, b)
p /= length(possible)
for a in possible
    if a == TERM
        v += p * term_reward(s)
    else
        rec(s, p)
    end
end
results(s, a)

