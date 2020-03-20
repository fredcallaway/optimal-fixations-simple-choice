using Distributed
@everywhere using Printf
@everywhere begin
    using Glob
    using Serialization
    using SplitApplyCombine


    include("fit_base.jl")
    include("compute_policies.jl")
    include("compute_likelihood.jl")

    PRM_PATH = "$BASE_DIR/retest/test_prms"
    mkpath(dirname(PRM_PATH))

    function loss(n_item, prm)
        policies, t1 = @timed compute_policies(n_item, prm)
        (logp, ε, baseline), t2 = @timed likelihood(policies, prm.β_μ; LIKELIHOOD_PARAMS..., fold=:train)
        @printf "%d items: policies %ds   likelihood %ds\n" n_item t1 t2
        logp
    end

    # function loss(prm)
    #     mapreduce(+, [2, 3]) do n_item
    #         loss(n_item, prm)
    #     end
    # end

    function compute_loss(job)
        prm = deserialize(PRM_PATH)[job]
        map([2,3]) do n_item
            loss(n_item, prm)
        end
    end
end


if ARGS[1] == "setup"
    TOP_N = parse(Int, ARGS[2])
    REPETITIONS = parse(Int, ARGS[3])

    results = map(glob("$BASE_DIR/likelihood/*")) do f
        endswith(f, "x") && return missing
        try
            deserialize(f)
        catch
            println("Can't read $f")
            missing
        end
    end |> skipmissing |> flatten

    prms, l2, l3, lc = map(results) do (prm, losses)
        prm, losses[1], losses[2], sum(losses)
    end |> invert;

    rank = sortperm(lc)
    top_prms = prms[rank[1:TOP_N]]
    test_prms = repeat(top_prms, REPETITIONS);
    serialize(PRM_PATH, test_prms)
    println("Wrote $(length(test_prms)) parameter configurations to $PRM_PATH")
else
    jobs = eval(Meta.parse(ARGS[1]))
    pmap(jobs) do job
        do_job(compute_loss, "retest", job)
    end
end

