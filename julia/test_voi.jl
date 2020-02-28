
let
    m = MetaMDP()

    beliefs = map(1:10000) do i
        b = Belief(m)
        b.μ .+= randn(3)
        b.λ .+= rand(3)
        b
    end

    n_sample = 100000
    n_belief = 1000
    @time x = map(1:n_belief) do i
        mean_mc, std_mc = vpi(b, n_sample)
        sem_mc = std_mc / √n_sample
        (vpi_clever(b) - mean_mc) / sem_mc
    end

    sem_x = 1 / √n_belief

    @assert -2sem_x < mean(x) < 2sem_x
    assert 0.8 < std(x) < 1.2
    @assert maximum(abs.(x)) < 5
end