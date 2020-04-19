using Distributed
using Serialization
using NPZ

@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("binning.jl")
end
policies = deserialize("results/main14/test_policies/joint-false/1");
pol2 = policies[1][1]
pol3 = policies[2][1]

@everywhere function sample_bs(pol, n_roll)
    bs = Belief[]
    for i in 1:n_roll
        rollout(pol, callback=(b,c)->push!(bs, deepcopy(b)))
    end
    bs
end

@everywhere function modify(μ, x)
    if length(μ) == 2
        μ = sort(μ; rev=true)
        μ[1] = μ[2] + x
        return μ
    end
    μ = sort(μ)
    if x < 0
        μ[1] = μ[2] + x
    elseif x > 0
        μ[1], μ[3] = μ[3], μ[1]
        μ[1] = μ[2] + x
    else
        μ[1], μ[2] = μ[2], μ[1]
    end
    μ
end

for b in sample_bs(pol3, 100)
    x = randn()
    μ = modify(b.μ, x)
    try
        @assert μ[1] - median(μ) ≈ x
    catch e
        error("failure $x $(b.μ) $μ")
    end
end


function make_X(pol, n_roll=N_ROLL)
    bs_ = map(sample_bs(pol, n_roll)) do b
        b.focused = -1
        b
    end
    @everywhere bs = $bs_  # pre-transfer bs to each process
    pmap(Iterators.product(μs, λs), batch_size=100) do (x, λ)
        length(bs) \ mapreduce(+, bs) do b
            old_μ = b.μ
            old_λ = b.λ[1]
            b.μ = modify(b.μ, x)
            b.λ[1] = λ
            v = [voc(pol, b); 0.]
            b.μ = old_μ
            b.λ[1] = old_λ
            softmax(pol.α .* v)[1]
        end
    end |> transpose |> collect
end

# μs = -1:0.005:1
# λs = 1:0.005:3
# N_ROLL = 5000
μs = -1:0.01:1
λs = 1:0.01:3
N_ROLL = 1000
dest = "results/big_policy_npy"

mkpath("dest")
npzwrite("$dest/2", make_X(pol2))
npzwrite("$dest/3", make_X(pol3))
npzwrite("$dest/μs", collect(μs))
npzwrite("$dest/λs", collect(λs))
