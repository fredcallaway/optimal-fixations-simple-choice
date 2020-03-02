using Serialization
using Sobol

include("box.jl")
include("meta_mdp.jl")
include("bmps.jl")

BASE_DIR = "results/big-grid"
TEST_RUN = false
SEARCH_STRATEGY = :grid

if basename(PROGRAM_FILE) == basename(@__FILE__)
    isfile("$BASE_DIR/run_params") && error("This run has already been set up.")
    mkpath(BASE_DIR)
    SPACE = Box(
        :α => (50, 500),
        :σ_obs => (1, 10),
        :sample_cost => (.001, .01),
        :switch_cost => (.003, .03),
    )
    # SPACE = Box(
    #     :α => (100, 300),
    #     :σ_obs => (2, 3.5),
    #     :sample_cost => (.001, .006),
    #     :switch_cost => (.013, .025),
    # )

    if TEST_RUN 
        UCB_PARAMS = (
            N=8,
            n_iter=2,
            n_init=2,
            n_roll=2,
            n_top=2,
        )
        LIKELIHOOD_PARAMS = (
            fit_ε = true,
            max_ε = 0.5,
            n_sim_hist = 10,
            test_fold = "odd",
            hist_bins = 5
        )
    else
        UCB_PARAMS = (
            n_iter=500,
            n_init=100,
            n_roll=100,
            n_top=80
        )
        LIKELIHOOD_PARAMS = (
            fit_ε = true,
            max_ε = 0.5,
            n_sim_hist = 10_000,
            test_fold = "odd",
            hist_bins = 5
        )
    end
    
    things = (SPACE, UCB_PARAMS, LIKELIHOOD_PARAMS)
    serialize("$BASE_DIR/run_params", things)
    println("Wrote $BASE_DIR/run_params")
    foreach(println, things)
else
    SPACE, UCB_PARAMS, LIKELIHOOD_PARAMS = deserialize("$BASE_DIR/run_params")
end


function get_prm(job)
    x = if SEARCH_STRATEGY == :sobol
        seq = SobolSeq(n_free(SPACE))
        skip(seq, job-1; exact=true)
        next!(seq)
    elseif SEARCH_STRATEGY == :grid
        g = range(0,1,length=10)
        mg = Iterators.product(repeat([g], n_free(SPACE))...)
        collect(collect(mg)[job])
    else
        error("Invalid search strategy: $SEARCH_STRATEGY")
    end

    x |> SPACE |> namedtuple
end

function do_job(f::Function, name::String, job::Int)
    out = "$BASE_DIR/$name"
    mkpath(out)
    dest = "$out/$job"
    if isfile(dest)
        println("$dest already exists")
        return
    elseif isfile(dest * "x")
        println("$dest is already in progress")
        return
    end
    touch(dest*"x")
    try
        println("Computing $dest"); flush(stdout)
        results = f(job)
        serialize(dest, results)
        println("Wrote $dest"); flush(stdout)
    finally
        rm(dest*"x")
    end
end




