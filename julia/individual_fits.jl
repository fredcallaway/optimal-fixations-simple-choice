using ProgressMeter
@everywhere include("fit_base.jl")
@everywhere include("pseudo_likelihood.jl")

path = "$BASE_DIR/individual"
mkpath(path)

function individual_like()
    individual_data = mapmany([2, 3]) do n_item
        collect(group(x->x.subject, load_dataset(n_item, :train)))
    end;

    results = @time @showprogress pmap(CachingPool(workers()), 1:10000) do job
        map(1:7) do prior_i
            # job = 1; prior_i = 1
            prm, both_histograms = deserialize("$BASE_DIR/likelihood/$prior_i/$job");
            map(individual_data) do trials
                n_item = length(trials[1].value)
                (n_item, trials[1].subject) => likelihood(trials, both_histograms[n_item-1])
            end |> Dict
        end
    end
    results |> flatten |> invert |> OrderedDict |> sort!
end

# %% ==================== Identify MLE ====================

like = individual_like()
serialize("$path/likelihood", like)
serialize("$path/keys", collect(keys(like)))

K = map(collect(keys(like))) do k
    fill(k, 30)
end |> combinedims
serialize("$path/K", K)

all_prms = deserialize("$BASE_DIR/all_prms");
all_top = map(collect(values(like))) do ll
    top = partialsortperm(first.(ll), 1:30; rev=true)
    all_prms[top]
end |> combinedims
serialize("$path/all_top", all_top)

# %% --------
map(eachcol(all_top)) do prms
    map(mean, invert(prms))
end


# %% ==================== Aggregate plot features and simulations ====================


let
    sims = Dict(k => Table[] for k in keys(like))
    for (i, (n_item, subject)) in enumerate(K)
        push!(sims[(n_item, subject)], reduce(vcat, deserialize("$path/simulations/$i")))
    end
    serialize("$path/processed/simulations", sims)
end


let
    feats = Dict(k => Dict[] for k in keys(like))
    for (i, (n_item, subject)) in enumerate(K)
        push!(feats[(n_item, subject)], deserialize("$path/plot_features/$i"))
    end
    serialize("$path/processed/plot_features", feats)
end

# %% --------
deserialize("$path/simulations/1000") |> values |> first
deserialize("$path/simulations/1001") |> values |> first


