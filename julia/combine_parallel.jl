include("results.jl")
all_res = filter(get_results("test")) do res
    exists(res, :args) || return false
    args = load(res, :args)
    args["bmps_iter"] == 500 &&
    args["fold"] == "even" &&
    exists(res, :loss)
end

all_res = unique(all_res) do res
    load(res, :args)["job"]
end

all_args = map(all_res) do res
    load(res, :args)
end

jobs = Int[]
u_args = unique(all_args) do args
    push!(jobs, pop!(args, "job"))
    args
end
@assert length(u_args) == 1
args = u_args[1]
res = Results("test_post")
include("fit_pseudo_base.jl")

# %% ====================  ====================
prms, losses = map(all_res) do res
    load(res, :prm), load(res, :loss)
end |> invert;

xs = map(prms) do prm
    space(type2dict(prm))
end

X = combinedims(xs)
y = losses
# %% ====================  ====================

# train = sample(eachindex(y), 800; replace=false)
train = eachindex(y)
# train = 1:512
# test = setdiff(eachindex(y), train)
opt = gp_minimize(loss, n_free(space);
    init_Xy=(X[:, train], y[train]), run=false, verbose=false, iterations=50)

# optimize!(opt.model);
# ℓ = -log.(opt.model.kernel.iℓ2) / 2 # log length scales
# find_model_max!(opt)

# prm = opt.model_optimizer |> space |> Params
# save(res, :mle, prm)
# loss(opt.model_optimizer)

fit(opt; n_iter=4)
# %% ====================  ====================
prm = opt.model_optimizer |> space |> Params


# %% ====================  ====================

d = n_free(space)
model = GP(X[:, train], y[train], MeanConst(0.), Mat32Ard(zeros(d), 5.), -2.)
optimize!(model)

yhat, yvar = predict_f(model, X[:, test]);


cor([y[test] yhat])

# %% ====================  ====================



# save(results, Symbol(string("mle_", loss_iter)), prm)
# save(results, :gp_model, opt.model)
ℓ = -log.(opt.model.kernel.iℓ2) / 2 # log length scales
loss = opt.model_optimum
@info "Iteration $loss_iter" loss prm repr(ℓ)

fit(opt)


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

