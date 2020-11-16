using ProgressMeter
@everywhere begin
    include("fit_base.jl")
    include("human.jl")
    include("pseudo_likelihood.jl")
end

mkpath("$BASE_DIR/likelihood/processed")

# %% --------

function get_likelihood(both_train)
    results = @time @showprogress pmap(1:10000) do job
        map(1:7) do prior_i
            prm, both_histograms = deserialize("$BASE_DIR/likelihood/$prior_i/$job");
            (prm=prm, like=map(likelihood, both_train, both_histograms))
        end
    end
    results |> flatten |> invert
end

both_train = map([2, 3]) do n_item
    filter(x->iseven(x.trial), load_dataset(n_item))
end
all_prm, like = get_likelihood(both_train)
# serialize("$BASE_DIR/all_prms", prms)

# %% --------

prior_ok(mode, β_μ) = Dict(
    :fit => true,
    :unbiased => β_μ == 1,
    :zero => β_μ == 0
)[mode]


function mean_std_str(k, xs, sigdigits=3)
    m, s = juxt(mean, std)(xs)
    s = @sprintf( "%-12s", k) *
    @sprintf("%8s", round(m; sigdigits=sigdigits)) *
    " ± " *
    @sprintf("%s", round(s; sigdigits=sigdigits))
    s
end

function write_fits(f, top, losses)
    for (k, xs) in pairs(type2dict(invert(top)))
        println(f, mean_std_str(k, xs))
    end
    println(f, mean_std_str("loss", losses, 5))
end

mkpath("$BASE_DIR/best_parameters/")
open("$BASE_DIR/best_parameters/mle.txt", "w+") do mle_file
    for prior_mode in [:fit, :unbiased, :zero]
        l2, l3, lc = map(prm, like) do prm, ll
            prior_ok(prior_mode, prm.β_μ) || return (Inf, Inf, Inf)
            l2 = -ll[1][1]
            l3 = -ll[2][1]
            lc = l2 + l3
            l2, l3, lc
        end |> collect |> invert;


        for (dataset, loss) in zip(["two", "three", "joint"], [l2, l3, lc])
            idx = partialsortperm(loss, 1:30)
            best = prm[idx]
            fp = "$BASE_DIR/best_parameters/$dataset-$prior_mode"
            serialize(fp, best)
            println("Wrote ", fp)
            println(mle_file, "----- dataset = $dataset  prior_mode = $prior_mode -----\n")
            # println(mle_file, join(idx, " "))
            write_fits(mle_file, best, partialsort(loss, 1:30))
            print(mle_file, "\n\n")
        end
    end
end
println("Wrote $BASE_DIR/best_parameters/mle.txt")
print(read("$BASE_DIR/best_parameters/mle.txt", String))

# %% --------
best = deserialize("$BASE_DIR/best_parameters/joint-zero")
rng = map(juxt(minimum, maximum), invert(best))
map(free(SPACE)) do k
    a, b = SPACE[k]
    x, y = getfield(rng, k)
    k => [x - a, b - y] ./ (b - a)
end




# %% --------
loss = map(like) do ll
    -(ll[1][1] + ll[2][1])
end

rank = sortperm(loss)
top = prm[rank[1:30]]
serialize("$BASE_DIR/best_parameters/joint-")

write_fits("$BASE_DIR/mle.txt", top, loss)
println(run(`cat "$BASE_DIR/mle.txt"`))


