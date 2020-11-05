using ProgressMeter
include("fit_base.jl")
include("pseudo_likelihood.jl")

# %% --------

function get_likelihood(both_train)
    results = @time @showprogress map(1:10000) do job
        map(1:7) do prior_i
            prm, both_histograms = deserialize("$BASE_DIR/likelihood/$prior_i/$job");
            (prm=prm, like=map(likelihood, both_train, both_histograms))
        end
    end
    results |> flatten |> invert
end

function compute_recovery_likelihood(job::Int, n_trial)
    prm, both_sims = deserialize("$BASE_DIR/recovery/sims/$job")
    if !ismissing(n_trial)
        both_sims = map(both_sims) do sims
            sims[1:n_trial]
        end
    end
    print(length.(both_sims))
    get_likelihood(both_sims)[2]
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    job = eval(Meta.parse(ARGS[1]))
    n_trial = length(ARGS) > 1 ? parse(Int, ARGS[2]) : missing
    path = ismissing(n_trial) ? "recovery/likelihood/full" : "recovery/likelihood/$n_trial"
    do_job(compute_recovery_likelihood, path, job, n_trial)
end
