using Distributed
@everywhere begin
    out = "results/bernoulli"
    using Serialization

    include("bernoulli_meta_mdp.jl")
    # include("optimize_bmps.jl")
    include("ucb_bmps.jl")

    UCB_PARAMS = (
        n_iter=500,
        n_init=100,
        n_roll=100,
        n_top=80
    )
    N_EVAL = 100_000
    MAX_OBS = 75

    jobs = Iterators.product(
        exp.(-9:-3),
        1:10,
    ) |> collect

    function do_job(i)
        println("do job $i")
        sample_cost, switch_cost = jobs[i]
        try
            m = MetaMDP(sample_cost=sample_cost, switch_cost=switch_cost, max_obs=MAX_OBS)
            V = ValueFunction(m)
            opt_val = V(Belief(m))
            ucb_out = ucb(m; UCB_PARAMS...)
            policies, μ, sem = ucb(m; UCB_PARAMS...)
            best = partialsortperm(-μ, 1:UCB_PARAMS.n_top)
            best_pols = policies[best]
            ucb_vals = map(best_pols) do pol
                N_EVAL \ mapreduce(+, 1:N_EVAL) do i
                    rollout(pol).reward
                end
            end

            # @info "Loss" sample_cost switch_cost bmps_val - opt_val bmps_val / opt_val

            serialize("$out/$i", (
                m=m,
                ucb=ucb_out,
                ucb_vals=ucb_vals,
                opt_val=opt_val,
            ))
            println("Wrote $out/$i")
        catch e
            println("Error processing ($sample_cost, $switch_cost)")
            println(e)
        end

    end

end # @everywhere
mkpath(out)
pmap(do_job, 1:70)





