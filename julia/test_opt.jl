cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
using Distributed
addprocs(48)
@everywhere include("model.jl")
include("job.jl")
using SplitApplyCombine
using Printf

function x2theta(x)
    cost_weight = x[1]
    voi_weights = diff([0; sort(collect(x[2:end])); 1])
    [cost_weight; voi_weights]
end

job = Job("runs/time8/jobs/2.json")
m = MetaMDP(job)

optim = load(job, :optim)
X = optim["X"]
y = Vector{Float64}(optim["y"])
sp = sortperm(y)

juxt(fs...) = (xs...) -> Tuple(f(xs...) for f in fs)
function evaluate(pol; n_roll=10000)
    e = pmap(1:n_roll) do i
        roll = rollout(pol; max_steps=1000)
        roll.reward, roll.steps
    end |> invert .|> juxt(mean, std)
    (rm, rs), (sm, ss) = e
    println(round.(pol.θ; digits=3))
    @printf "  Reward = %.2f ± %.2f;  Steps = %.2f ± %.2f\n" rm rs sm ss
    (reward=(mean=rm, std=rs), steps=(mean=sm, std=ss))
end

N = 10000

pol1 = Policy(m, x2theta(optim["x1"]))
r1 = evaluate(pol1, n_roll=N)
@printf "  Expected Reward = %.2f\n" -optim["y1"]
results = map(sp) do i
    x = X[i]
    pol = Policy(m, x2theta(x))
    evaluate(pol)
    @printf "  Expected Reward = %.2f\n" -y[i]
end
using Serialization
open("test_opt_results.jls", "w+") do f
    serialize(f, [r1; results])
end
# open(deserialize, "test_opt_results.jls")
