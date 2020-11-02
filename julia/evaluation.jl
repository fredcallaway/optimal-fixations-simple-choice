@everywhere begin
    include("fit_base.jl")
    include("compute_policies.jl")
    include("human.jl")
    include("simulations.jl")
    include("pseudo_likelihood.jl")
    include("plots_preprocessing.jl")

    function get_top_prm(job::Int, n_item::Int)
        # return deserialize("$BASE_DIR/retest/top30")[job]

        if DATASET == "joint"
             return deserialize("$BASE_DIR/best_parameters/joint-$PRIOR_MODE")[job]
        elseif DATASET == "separate"
            num = Dict(2 => "two", 3 => "three")[n_item]
            return deserialize("$BASE_DIR/best_parameters/$num-$PRIOR_MODE")[job]
        end
        error("No parameter found!")
    end

    function recompute_policies(job::Int)
        map([2,3]) do n_item
            prm = get_top_prm(job, n_item)
            compute_policies(n_item, prm; UCB_PARAMS...)
        end
    end

    function compute_simulations(job::Int)
        both_policies = deserialize("$BASE_DIR/test_policies/$DATASET-$PRIOR_MODE/$job")
        map([2,3], both_policies) do n_item, policies
            prm = get_top_prm(job, n_item)
            prior = make_prior(load_dataset(n_item), prm.β_μ)
            trials = load_dataset(n_item, :test)
            map(policies) do pol
                simulate_trials(pol, prior, trials)
            end
        end
    end

    function compute_test_likelihood(job::Int)
        both_policies = deserialize("$BASE_DIR/test_policies/$DATASET-$PRIOR_MODE/$job")
        map([2,3], both_policies) do n_item, policies
            prm = get_top_prm(job, n_item)
            likelihood(policies, prm.β_μ; LIKELIHOOD_PARAMS..., fold=:test)
        end
    end

    function compute_plot_features(job::Int)
        sims = map(deserialize("$BASE_DIR/simulations/$DATASET-$PRIOR_MODE/$job")) do ss
            # combine all the policy simulations into one big table
            reduce(vcat, ss)
        end
        compute_plot_features(sims)
    end
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    DATASET = ARGS[1]
    PRIOR_MODE = ARGS[2]
    @everywhere DATASET = $DATASET
    @everywhere PRIOR_MODE = $PRIOR_MODE

    N = length(deserialize("$BASE_DIR/best_parameters/$DATASET-$PRIOR_MODE"))
    pmap(1:N) do job
        do_job(recompute_policies, "test_policies/$DATASET-$PRIOR_MODE", job)
        do_job(compute_simulations, "simulations/$DATASET-$PRIOR_MODE", job)
        do_job(compute_plot_features, "plot_features/$DATASET-$PRIOR_MODE", job)
        # do_job(compute_test_likelihood, "test_likelihood/$DATASET-$PRIOR_MODE", job)
    end
end

#=
julia -p 30 evaluation.jl joint true &
julia -p 30 evaluation.jl joint false &
julia -p 30 evaluation.jl separate true &
julia -p 30 evaluation.jl separate false &
=#
