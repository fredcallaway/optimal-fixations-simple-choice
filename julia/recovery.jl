include("fit_base.jl")
using CSV

out = "$BASE_DIR/recovery/results"
mkpath(out)

all_prm = deserialize("$BASE_DIR/all_prm")
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
        (top1=all_prm[best[1]], top30=map(mean, (invert(all_prm[best]))))
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
both_results = deserialize("tmp/recovery_both_results");
Rs = map(Iterators.product([2, 3], [:group, :indiv])) do (n_item, meth)
    results = combinedims(both_results[n_item-1]);
    R = @. first(getfield(results, meth))
    mle = [x.I[2] for x in argmax(R; dims=2)][:]
    prms = first.(all_histograms[mle])
    prms |> Table |> CSV.write("$out/mle-$n_item-$meth")
    map(enumerate(eachrow(R))) do (i, r)
        rank = sortperm(sortperm(-r))
        (rank=rank[to_sim[i]],)
    end |> CSV.write("$out/rank-$n_item-$meth")
    R
end;
# %% --------

Rs = map(both_results) do res
    results = combinedims(res);
    @. first(getfield(results, :group))
end

R = Rs[1] .+ Rs[2];
mle = [x.I[2] for x in argmax(R; dims=2)][:]
prms = first.(all_histograms[mle])
prms |> Table |> CSV.write("$out/mle-joint")
map(enumerate(eachrow(R))) do (i, r)
    rank = sortperm(sortperm(-r))
    (rank=rank[to_sim[i]],)
end |> CSV.write("$out/rank-joint")

# %% --------

R = Rs[2,1]
r = R[4, :]



rank = sortperm()

map(eachrow(R)) do r


# %% --------

using CSV
out = "$BASE_DIR/recovery"
mkpath(out)

true_prms = first.(all_histograms[to_sim])


true_prms |> Table |> CSV.write("$out/true_prms")
mle_params(group) |> Table |> CSV.write("$out/mle_group")
mle_params(indiv) |> Table |> CSV.write("$out/mle_indiv")

# %% --------
# print(join([x.I[2] for x in argmax(R; dims=2)], " "))



top = sortperm(-like)[1:5]
map(top) do t
    all_histograms[t][1]
end
prm



map(policies) do pol
    simulate_trials(pol, prior, trials)
end