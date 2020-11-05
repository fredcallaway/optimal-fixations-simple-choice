include("fit_base.jl")
using CSV

out = "$BASE_DIR/recovery/results"
mkpath(out)

all_prms = deserialize("$BASE_DIR/all_prms");
true_prms = deserialize("$BASE_DIR/recovery/true_prms")

all_like_full = asyncmap(1:1024) do job
    deserialize("$BASE_DIR/recovery/likelihood/full/$job")
end;
all_like_50 = asyncmap(1:1024) do job
    deserialize("$BASE_DIR/recovery/likelihood/50/$job")
end;
# %% --------
function get_mle(all_like)
    map(all_like) do like
        loss = map(like) do ll
            -(ll[1][1] + ll[2][1])
        end
        best = partialsortperm(loss, 1:30)
        (top1=all_prms[best[1]], top30=map(mean, (invert(all_prms[best]))))
    end |> invert
end

true_prms |> CSV.write("$out/true.csv")

top1, top30 = get_mle(all_like_full)
top1 |> CSV.write("$out/top1.csv")
top30 |> CSV.write("$out/top30.csv")

top1, top30 = get_mle(all_like_50)
top1 |> CSV.write("$out/top1-50.csv")
top30 |> CSV.write("$out/top30-50.csv")

# Fin!

# %% --------

intersect(dropnames.(true_prms, :β_μ), dropnames.(all_prms, :β_μ))
mean(dropnames.(true_prms, :β_μ) .== dropnames.(top1, :β_μ))


top1
