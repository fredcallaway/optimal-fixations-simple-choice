include("results.jl")
include("pseudo_likelihood.jl")
include("box.jl")

# %% ==================== Choose result ====================
results = filter(get_results("pseudo_mu_cv")) do res
    exists(res, :like_kws) &&
    !load(res, :like_kws).fit_ε &&
    exists(res, :loss_400)
end


# res = get_result("results/pseudo_mu_cv/2019-10-07T22-07-13-6yh/")
res = results[1]
# %% ====================  ====================

using Printf
losses = []
for i in 1:800
    try
        push!(losses, load(res, Symbol(@sprintf("loss_%03d", i))))
    catch
        break
    end
end

space = load(res, :space)

# %% ====================  ====================

xs, y = map(losses) do loss
    x = space(type2dict(loss.prm))
    y = loss.likelihood / BASELINE
    x, y
end |> invert



# %% ====================  ====================
d = n_free(space)
d = 2
model = ElasticGPE(d,
  mean = MeanConst(0.),
  kernel = Mat32Ard(zeros(d), 5.),
  logNoise = -2.,
  capacity = length(xs)
)

model_optimizer = MAPGPOptimizer(
    every = 20,
    noisebounds = [-4, 5],
    maxeval = 200
)
function tell!(x, y; opt=true)
    append!(model, reshape(x, (:, 1)), [y])
    if opt
        BayesianOptimization.optimizemodel!(model_optimizer, model)
    end
end


X = rand(d, 200)
xs = splitdims(X)
y = sum(X; dims=1)[:]
# y = X[1, :] .+ X[2, :] .+ X[3, :] .+ X[4, :]

for i in 1:100
    tell!(xs[i], y[i]; opt=false)
end


preds = map(101:length(y)) do i
    print('.')
    yhat, yvar = predict_f(model, reshape(xs[i], (:, 1)))
    (xs[i], y[i], yhat[1], √yvar[1])
end

open("tmp/preds", "w") do f
    serialize(f, preds)
end













# %% ====================  ====================
d = 2
model = ElasticGPE(d,
  mean = MeanConst(0.),
  kernel = Mat32Ard(zeros(d), 5.),
  logNoise = -2.,
  capacity = 1000
)

model_optimizer = MAPGPOptimizer(
    every = 20,
    noisebounds = [-4, 5],
    maxeval = 200
)

X = rand(2, 100)
xs = splitdims(X)
y = X[1, :] .+ X[2, :]

# model = GP(xs, y, MeanConst(0.), Mat32Ard(zeros(d), 5.), -2.);


for i in 1:100
    append!(model, reshape(xs[i], (:, 1)), [y[i]])
end

X = rand(2, 100)
y = X[1, :] .+ X[2, :]
yhat = predict_f(model, X)
open("tmp/easy", "w") do f
    serialize(f, (y, yhat))
end


