include("results.jl")
include("pseudo_likelihood.jl")
include("box.jl")

all_res = filter(get_results("both_items_fixed_parallel")) do res
    exists(res, :loss)
end

prms, losses = map(all_res) do res
    load(res, :prm), load(res, :loss)
end |> invert;


space = Box(
    :sample_time => 100,
    :α => (50, 300, :log),
    :σ_obs => (1, 6),
    :sample_cost => (.001, .01, :log),
    :switch_cost => (.01, .05, :log),
    :σ_rating => 0.,
    # :µ => args["fit_mu"] ? (0, μ_emp) : μ_emp,
    :μ => (0,5),
    :σ => 2.55,  # FIXME
)

# %% ====================  ====================

xs = map(prms) do prm
    space(type2dict(prm))
end

X = combinedims(xs)
y = losses

opt = gp_minimize(x->0, n_free(space);
    init_Xy=(X, y),
    run=false, verbose=false)

optimize!(opt.model)
y1, x1 = find_model_max!(opt)
space(x1)

# %% ====================  ====================



d = n_free(space)
model = GP(X, y, MeanConst(0.), Mat32Ard(zeros(d), 5.), -2.)
optimize!(model);


# %% ====================  ====================
test = 1:600
train = setdiff(eachindex(y), test);

model = GP(X[:, train], y[train], MeanConst(0.), Mat32Ard(zeros(d), 5.), -2.)
optimize!(model);

