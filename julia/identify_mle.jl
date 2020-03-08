using Glob
using SplitApplyCombine
using Serialization

BASE_DIR = "results/$(ARGS[1])"

function identify_mle()
    results = map(glob("$BASE_DIR/likelihood/*")) do f
        endswith(f, "x") && return missing
        try
            deserialize(f)
        catch
            println("Can't read $f")
            missing
        end
    end |> skipmissing |> collect |> flatten

    mkpath("$BASE_DIR/best_parameters/")
    
    for fit_prior in (true, false)
        prms, l2, l3, lc = map(results) do (prm, losses)
            if !fit_prior
                prm.β_μ < 0.9 && return missing
                prm = (prm..., β_μ = 1)
            end
            prm, losses[1], losses[2], sum(losses)
        end |> skipmissing |> collect |> invert;

        for (num, loss) in zip(["two", "three", "joint"], [l2, l3, lc])
            best = prms[partialsortperm(loss, 1:30)]
            fp = "$BASE_DIR/best_parameters/$num-$fit_prior"
            serialize(fp, best)
            println("Wrote ", fp)
        end
    end
end
identify_mle()