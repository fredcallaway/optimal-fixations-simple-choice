using Distributed

repetition = ARGS[1]

@everywhere using Printf
@everywhere begin
    using Glob
    using Serialization
    using SplitApplyCombine

    include("fit_base.jl")
    include("compute_policies.jl")
    include("compute_likelihood.jl")

    FIT_MODE = "joint"
    FIT_PRIOR = false
    repetition = $repetition
    RES_PATH = "$BASE_DIR/retest/$FIT_MODE-$FIT_PRIOR/$repetition"

    # RETEST_UCB_PARAMS = (
    #     N=5^3,
    #     n_iter=2,
    #     n_init=2,
    #     n_roll=2,
    #     n_top=5,
    # )
    # RETEST_LIKELIHOOD_PARAMS = (
    #     fit_ε = true,
    #     max_ε = 0.5,
    #     n_sim_hist = 10,
    #     test_fold = "odd",
    #     hist_bins = 5
    # )

    function loss(n_item, prm)
        policies, t1 = @timed compute_policies(n_item, prm; RETEST_UCB_PARAMS...)
        (logp, ε, baseline), t2 = @timed likelihood(policies, prm.β_μ; RETEST_LIKELIHOOD_PARAMS..., fold=:train)
        @printf "%d items: policies %ds   likelihood %ds\n" n_item t1 t2
        -logp
    end

    function loss(prm)
        if FIT_MODE == "two"
            loss(2, prm)
        elseif FIT_MODE == "three"
            loss(3, prm)
        elseif FIT_MODE == "joint"
            loss(2, prm) + loss(3, prm)
        else
            error("Bad FIT_MODE")
        end
    end
end

function choose_loss(losses)
    if FIT_MODE == "two"
        losses[1]
    elseif FIT_MODE == "three"
        losses[2]
    elseif FIT_MODE == "joint"
        sum(losses)
    else
        error("Bad FIT_MODE")
    end
end

function identify_top_prms()
    results = map(glob("$BASE_DIR/likelihood/*")) do f
        endswith(f, "x") && return missing
        try
            deserialize(f)
        catch
            println("Can't read $f")
            missing
        end
    end |> skipmissing |> collect;

    prms, losses = map(results) do ress
        p, l = map(ress) do (prm, losses)
            prm, choose_loss(losses)
        end |> invert
        i = argmin(l)
        p[i], l[i]
    end |> invert

    best = minimum(losses)
    threshold = best + 200.  # maximum observed normalized difference from intense likelihood
    keep = losses .< threshold
    # rank = sortperm(losses)
    # keep = rank[1:96]
    prms[keep]
    zip(prms[keep], losses[keep])
end

mkpath(RES_PATH)
top_prms = collect(identify_top_prms());
pmap(enumerate(top_prms)) do (i, (prm, old_loss))
    println("Computing retest loss for job $i")
    serialize("$RES_PATH/$i", (
        prm=prm,
        loss=loss(prm),
        old_loss=old_loss,
    ))
end


# # %% ====================  ====================
# if ARGS[1] == "setup"
#     TOP_N = parse(Int, ARGS[2])
#     REPETITIONS = parse(Int, ARGS[3])

#     results = map(glob("$BASE_DIR/likelihood/*")) do f
#         endswith(f, "x") && return missing
#         try
#             deserialize(f)
#         catch
#             println("Can't read $f")
#             missing
#         end
#     end |> skipmissing |> flatten

#     prms, l2, l3, lc = map(results) do (prm, losses)
#         prm, losses[1], losses[2], sum(losses)
#     end |> invert;

#     rank = sortperm(lc)
#     top_prms = prms[rank[1:TOP_N]]
#     test_prms = repeat(top_prms, REPETITIONS);
#     serialize(PRM_PATH, test_prms)
#     println("Wrote $(length(test_prms)) parameter configurations to $PRM_PATH")
# else
#     jobs = eval(Meta.parse(ARGS[1]))
#     pmap(jobs) do job
#         do_job(compute_loss, "retest", job)
#     end
# end

