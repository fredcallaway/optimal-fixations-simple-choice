include("model.jl")
using BenchmarkTools
using SplitApplyCombine


m = MetaMDP(3, 7, 0.002, 8)
θ = [0.05, 0.8, 0, 0.2]
policy = SlowPolicy(m, θ)
s = State(m)
b = Belief(s)
@timed(mean(rollout(m, policy; state=s).steps for i in 1:100))[1:2]

# %% ==================== Profiling ====================

using Profile
Profile.init(delay=0.01)
Profile.clear()
f() = [rollout(m, policy; state=s) for i in 1:1000]
@profile f();
Profile.print(noisefloor=2)

Profile.print(format=:flat, sortedby=:count, noisefloor=2)
# %% ==================== VPI ====================
function vpi(b::Belief; n_sample=5000)
    s = b.mu .+ (b.lam .^ -0.5) .* mem_randn(length(b.mu), n_sample)
    mean(maximum(s; dims=1)) - maximum(b.mu)
end

function vpi2(b; n_sample=5000)
    # Use pre-allocated arrays efficiency
    R = mem_zeros(n_sample, length(b.mu))
    max_samples = mem_zeros(n_sample)

    R[:] = mem_randn(n_sample, length(b.mu))
    R .*= (b.lam .^ -0.5)' .+ b.mu'

    maximum!(max_samples, R)
    mean(max_samples) - maximum(b.mu)
end


@code_warntype vpi(b)
@time vpi(b)  # 213.199 μs (27 allocations: 156.92 KiB)
@time vpi2(b)  # 43.916 μs (14 allocations: 544 bytes)


x = [1]
x .*= 2 .+ 1
x .* 2 .+ 1

maxs = zeros(1, 5000)
@time maximum(s; dims=1);
@time maximum(st; dims=2);
@time maximum!(max_samples, st);
@time maximum!(maxss, s);

const maxss = zeros(1, 5000)


@time maximum!(max_samples, s)
@time maximum(s; dims=1)
@time vpi(b)

f() = @time begin
    X = mem_randn(length(b.mu), 5000)
    for i in 1:5000
        # view(X, :, 1)
        X[:, 1]
    end
end
maximum

@btime vpi2(b)


@btime vpi(b)
# %% ====================  ====================
function foo(π)
    x = Float64[]
    y = Float64[]
    rollout(m, smart) do b, c
        push!(x, vpi(b))
        voc1 = [voi1(b, c) - cost(π.m, b, c) for c in 1:π.m.n_arm]
        push!(y, maximum(voc1))
    end
    x, y
end

x, y = foo(smart)
plot([x y * 10])
hline!([0], c=:black)
