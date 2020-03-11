using Glob
using SplitApplyCombine
using Serialization

include("fit_base.jl")
# BASE_DIR = "results/$(ARGS[1])"

function mean_std_str(k, xs, sigdigits=3)
    m, s = juxt(mean, std)(xs)
    s = @sprintf( "%-12s", k) *
    @sprintf("%8s", round(m; sigdigits=sigdigits)) *
    " ± " *
    @sprintf("%s", round(s; sigdigits=sigdigits))
    s
end

function write_fits(f, best, losses)
    for (k, xs) in pairs(type2dict(invert(best)))
        println(f, mean_std_str(k, xs))
    end
    println(f, mean_std_str("loss", losses, 5))
end

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
    open("$BASE_DIR/best_parameters/mle.txt", "w+") do mle_file
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
                println(mle_file, "----- dataset = $num  fit_prior = $fit_prior -----\n")
                write_fits(mle_file, best, partialsort(loss, 1:30))
                print(mle_file, "\n\n")
            end
        end
    end
end
identify_mle()
