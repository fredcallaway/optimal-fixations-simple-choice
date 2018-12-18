using Test
include("model.jl")

const N_PROBLEM = 5
const N_SAMPLE = 10000
const N_ARM = 3
const m = MetaMDP(n_arm=N_ARM)
#%% ========== Helpers ==========
rand_state() = State(MetaMDP(
    n_arm = N_ARM,
    obs_sigma = 0.1 + 5rand()
))

function rand_belief()
    s = rand_state()
    b = Belief(s)
    while rand() < 0.9
        c = rand(1:N_ARM)
        step!(m, b, s, c)
    end
    b
end

"Estimates the value of f() ± 3*SEM"
function mc_est(f::Function; n_sample=N_SAMPLE)
    samples = [f() for i in 1:n_sample]
    sem = std(samples) / n_sample
    return mean(samples), max(1e-5, 3sem)
end


#%% ========== Tests ==========

@testset "voi1" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        while isempty(unobserved(b))
            b = rand_belief()
        end
        c = rand(unobserved(b))
        @test !observed(b, c)
        base = term_reward(b)
        mcq = mean(term_reward(observe(b, c)) for i in 1:N_SAMPLE)
        v = voi1(b, c)
        @test v >= 0
        @test mcq - base ≈ v atol=0.01
    end
    # b = Belief(rand_state())
    # observe!(b, 1)
    # voi1(b, 1) == 0
end


#%%

@testset "vpi" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        base = term_reward(b)
        est, ε = mc_est(()-> term_reward(observe_all(b)) - base)
        @test est ≈ vpi(b) atol=ε
    end
end


@testset "voi_gamble" begin
    for i in 1:N_PROBLEM
        # b = rand_belief()
        b = rand_belief()
        base = term_reward(b)
        g = rand(1:N_GAMBLE)
        est, ε = mc_est(()-> term_reward(observe_gamble(b, g)) - base)
        @test est ≈ voi_gamble(b, g) atol=ε
    end
end

@testset "voi_outcome" begin
    for i in 1:N_PROBLEM
        # b = rand_belief()
        b = rand_belief()
        base = term_reward(b)
        o = rand(1:N_OUTCOME)
        est, ε = mc_est(()-> term_reward(observe_outcome(b, o)) - base)
        @test est ≈ voi_outcome(b, o) atol=ε
    end
end

@testset "features" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        φ = features(b)
        int, v1, vg, vo, vp = [reshape(φ[i, :], N_OUTCOME, N_GAMBLE) for i in 1:5]
        unobs = .!(getfield.(b.matrix, :σ) .== 1e-20)
        @test all(int[unobs] .== -1)
        @test length(unique(vp[unobs])) <= 1
        @test map(1:N_GAMBLE) do i
            all(length(unique([-1e10; vg[:, i]])) <= 2)
        end |> all
    end
end

@testset "policy" begin
    pol = Policy([0; ones(4) ./ 4])
    rollout(pol, rand_state())

#%% ========== Scratch ==========

# @testset "meta greedy" begin
#     meta_greedy = Policy([0., 1., 0, 0, 0])
#     for i in 1:N_PROBLEM
#         b = rand_belief()
#         est, ε = mc_est(()-> rollout(meta_greedy, b).reward; n_sample=1000)
#         @test est >= (term_reward(b) - ε)
#     end
# end
