@everywhere begin
    include("fit_base.jl")
    include("compute_policies.jl")
    include("human.jl")
    include("simulations.jl")
    include("pseudo_likelihood.jl")
    include("plots_preprocessing.jl")

    function get_top_prm(job::Int, n_item::Int)
        # return deserialize("$BASE_DIR/retest/top30")[job]

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
            compute_policies(n_item, prm; UCB_PARAMS...)
        end
    end

    function compute_simulations(job::Int)
        both_policies = deserialize("$BASE_DIR/test_policies/$FIT_MODE-$FIT_PRIOR/$job")
        map([2,3], both_policies) do n_item, policies
            prm = get_top_prm(job, n_item)
            all_trials = load_dataset(policies[1].m.n_arm)
            prior = make_prior(all_trials, prm.β_μ)
            trials = get_fold(all_trials, LIKELIHOOD_PARAMS.test_fold, :test)
            map(policies) do pol
                simulate_trials(pol, prior, trials)
            end
        end
    end

    function compute_test_likelihood(job::Int)
        both_policies = deserialize("$BASE_DIR/test_policies/$FIT_MODE-$FIT_PRIOR/$job")
        map([2,3], both_policies) do n_item, policies
            prm = get_top_prm(job, n_item)
            likelihood(policies, prm.β_μ; LIKELIHOOD_PARAMS..., fold=:test)
        end
    end

    function compute_plot_features(job::Int)
        sims = map(deserialize("$BASE_DIR/simulations/$FIT_MODE-$FIT_PRIOR/$job")) do ss
            # combine all the policy simulations into one big table
            reduce(vcat, ss)
        end
        Dict(
            precompute(value_choice, sims; bin_spec=Binning(-4.5:1:4.5)),
            precompute(difference_time, sims),
            precompute(nfix_hist, sims),
            precompute(difference_nfix, sims),
            precompute(binned_fixation_times, sims),
            precompute(fixate_by_uncertain, sims; bin_spec=Binning(-50:100:850), three_only=true),
            precompute(value_bias, sims),
            precompute(value_duration, sims; fix_select=firstfix),
            precompute(fixate_on_worst, sims; cutoff=2000, n_bin=20),
            precompute(fix4_value, sims; three_only=true),
            precompute(fix3_value, sims; three_only=true),
            precompute(last_fix_bias, sims),
            precompute(fixation_bias, sims; bin_spec=7),
            precompute(first_fixation_duration, sims; bin_spec=7),
            precompute(first_fixation_duration_corrected, sims; bin_spec=7),
        )
    end
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    run_name = ARGS[1]
    FIT_MODE = ARGS[2]
    FIT_PRIOR = eval(Meta.parse(ARGS[3]))
    @everywhere run_name = $run_name
    @everywhere include("runs/$run_name.jl")
    @everywhere FIT_MODE = $FIT_MODE
    @everywhere FIT_PRIOR = $FIT_PRIOR

    N = length(deserialize("$BASE_DIR/best_parameters/$FIT_MODE-$FIT_PRIOR"))
    pmap(1:N) do job
        do_job(recompute_policies, "test_policies/$FIT_MODE-$FIT_PRIOR", job)
        do_job(compute_simulations, "simulations/$FIT_MODE-$FIT_PRIOR", job)
        do_job(compute_plot_features, "plot_features/$FIT_MODE-$FIT_PRIOR", job)
        do_job(compute_test_likelihood, "test_likelihood/$FIT_MODE-$FIT_PRIOR", job)
    end
end

#=
julia -p 30 evaluation.jl joint true &
julia -p 30 evaluation.jl joint false &
julia -p 30 evaluation.jl separate true &
julia -p 30 evaluation.jl separate false &
=#
