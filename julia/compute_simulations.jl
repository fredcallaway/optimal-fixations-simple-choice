@everywhere begin
    include("fit_base.jl")
    include("compute_policies.jl")
    include("human.jl")
    include("simulations.jl")

    function get_top_prm(job::Int, n_item::Int)
        if FIT_MODE == "joint"
             return deserialize("$BASE_DIR/best_parameters/joint-$FIT_PRIOR")[job]
        elseif FIT_MODE == "separate"
            num = Dict(2 => "two", 3 => "three")[n_item]
            return deserialize("$BASE_DIR/best_parameters/$num-$FIT_PRIOR")[job]
        end
        error("No parameter found!")
    end

    function recompute_policies(job::Int)
        map([2,3]) do n_item
            prm = get_top_prm(job, n_item)
            compute_policies(n_item, prm)
        end
    end

    function compute_simulations(job::Int)
        both_policies = deserialize("$BASE_DIR/simulation_policies/$FIT_MODE-$FIT_PRIOR/$job")
        map([2,3], both_policies) do n_item, policies
            prm = get_top_prm(job, n_item)
            trials = get_fold(load_dataset(n_item), LIKELIHOOD_PARAMS.test_fold, :test)
            prior = make_prior(trials, prm.β_μ)
            map(policies) do pol # =========================== FIME!!!
                simulate_trials(pol, prior, trials)
            end
        end
    end
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    FIT_MODE = ARGS[1]
    FIT_PRIOR = eval(Meta.parse(ARGS[2]))
    @everywhere FIT_MODE = $FIT_MODE
    @everywhere FIT_PRIOR = $FIT_PRIOR

    pmap(1:30) do job
        do_job(recompute_policies, "simulation_policies/$FIT_MODE-$FIT_PRIOR", job)
        do_job(compute_simulations, "simulations/$FIT_MODE-$FIT_PRIOR", job)
    end
end
