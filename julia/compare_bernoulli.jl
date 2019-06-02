using Distributed
@everywhere begin

using Serialization
include("optimize_bern.jl")
function expectation(f, n)
    acc = 0.
    for i in 1:n
        acc += f()
    end
    acc / n
end

function n_steps(pol)
    expectation(5000) do
        rollout(pol).steps
    end
end

function job((sample_cost, switch_cost))
    try
        m = MetaMDP(sample_cost=sample_cost, switch_cost=switch_cost, max_obs=50)
        # println("Solving meta-MDP")
        V = ValueFunction(m)
        v = V(Belief(m))
        println("Optimal value: $v")
        opt_steps = n_steps(OptimalPolicy(V))

        # println("Optimizing BMPS")
        bmps = optimize(m; verbose=false)
        pol = BMPSPolicy(m, bmps.θ1)
        bmps_steps = n_steps(pol)
        bmps_v = expectation(5000) do
            rollout(pol).reward
        end

        id = round(Int, rand() * 1e8)
        file = "bernoulli/$id"
        open(file, "w+") do f
            serialize(f, (
                m=m,
                bmps_result=bmps,
                opt_steps=opt_steps,
                bmps_steps=bmps_steps,
                bmps_v=bmps_v,
                opt_val=v,
                time_stamp=now()
            ))
        end
        println("Saved results to $file")
    catch
        println("Error processing ($sample_cost, $switch_cost)")
    end
end

end # @everywhere

args = Iterators.product(
    exp.(-9:-5),
    [1, 2, 3, 4, 5]
)
pmap(job, args)





