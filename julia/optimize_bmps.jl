using Distributed
using SharedArrays
using Printf

function max_cost(m::MetaMDP)
    θ = [1., 0, 0, 1]
    s = State(m)
    b = Belief(s)
    computes() = BMPSPolicy(m, θ)(b) != TERM

    while computes()
        θ[1] *= 2
    end

    while !computes()
        θ[1] /= 2
        if θ[1] < 2^-10
            error("Computation is too expensive")
        end
    end

    step_size = θ[1] / 100
    while computes()
        θ[1] += step_size
    end
    θ[1]
end

function halving(m::MetaMDP; N::Int=2^14, n_iter::Int=9, init_eval::Int=100, reduction=2)
    @debug "Begin successive halving" m N n_iter init_eval reduction
    mc = max_cost(m)

    function x2theta(x)
        voi_weights = diff([0; sort(collect(x[2:3])); 1])
        [x[1] * mc; voi_weights]
    end

    policies = [BMPSPolicy(m, x2theta(rand(3))) for i in 1:N]
    sort!(policies, by=x->-x.θ.vpi)  # for parallel efficiency

    n = SharedArray{Int}(N)
    v = SharedArray{Float64}(N)
    active = trues(N)
    fitness = zeros(N)

    n_eval = init_eval
    q = 1 // reduction
    n_active = N

    for iter in 1:n_iter
        t = @elapsed @sync @distributed for i in 1:N
            if active[i]
                policy = policies[i]
                v[i] += @distributed (+) for _ in 1:n_eval
                    rollout(policy, max_steps=200).reward
                end
                n[i] += n_eval
            end
        end

        fitness .= v ./ n
        threshold = quantile(fitness, 1-q)
        n_active = mapreduce(+, 1:N) do i
            active[i] = fitness[i] > threshold
        end

        @debug "iteration $iter" best=maximum(fitness) n_active q threshold n_eval time=t
        n_eval *= reduction
        q /= reduction
    end
    policies, fitness, n
end

function optimize_bmps(m::MetaMDP; kws...)
    policies, fitness, n = halving(m; kws...)
    fit, i = findmax(fitness)
    policies[i], fit
end

