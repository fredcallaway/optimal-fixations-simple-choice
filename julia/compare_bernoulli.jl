using Distributed
@everywhere begin

using Serialization
include("bernoulli_metabandits.jl")
include("optimize_bmps.jl")
include("results.jl")

function job((sample_cost, switch_cost))
    try
        m = MetaMDP(sample_cost=sample_cost, switch_cost=switch_cost, max_obs=50)
        V = ValueFunction(m)
        opt_val = V(Belief(m))
        bmps_pol, bmps_val = optimize_bmps(m)
        @info "Loss" sample_cost switch_cost bmps_val - opt_val bmps_val / opt_val

        id = round(Int, rand() * 1e8)
        res = Results("bernoulli")
        out = (
            m=m,
            bmps_val=bmps_val,
            opt_val=opt_val,
        )
        save(res, :out, out)
    catch e
        println("Error processing ($sample_cost, $switch_cost)")
        println(e)
    end
end

end # @everywhere

args = Iterators.product(
    exp.(-9:-3),
    1:10
)
pmap(job, args)





