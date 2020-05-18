@everywhere begin
    out = "results/bernoulli-may15"
    using Serialization

    include("bernoulli_meta_mdp.jl")
    include("bmps_ucb.jl")

    UCB_PARAMS = (
        N = 20^3,
        n_iter=5000,
        n_init=100,
        n_roll=10,
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
            policies, μ, sem = ucb_policies(m; UCB_PARAMS...)
            best = partialsortperm(-μ, 1:UCB_PARAMS.n_top)
            top_policies = policies[best]

            ucb_vals = map(top_policies) do pol
                N_EVAL \ mapreduce(+, 1:N_EVAL) do i
                    rollout(pol).reward
                end
            end

            V = ValueFunction(m)
            opt_val = V(Belief(m))

            # @info "Loss" sample_cost switch_cost bmps_val - opt_val bmps_val / opt_val

            serialize("$out/$i", (
                m=m,
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





