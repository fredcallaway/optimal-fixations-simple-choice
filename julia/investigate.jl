# include("inference.j")
include("elastic.jl")

if get(ARGS, 1, "") != "worker"
    include("results.jl")
    include("box.jl")
    addprocs(topology=:master_worker)
    # addprocs([("griffiths-gpu01.pni.princeton.edu", :auto)], tunnel=true, topology=:master_worker)
    println(nprocs(), " processes")
    results = Results("investigate/100")
end

@everywhere begin
    const REWEIGHT = false
    const N_PARTICLE = 1000
    include("inference.jl")
    const data = Datum.(trials, 100)
end


# %% ====================  ====================


if get(ARGS, 1, "") == "worker"
    start_worker()
else
    if get(ARGS, 1, "") == "master"
        start_master(wait=false)
    end

    save(results, :config, (
        reweight = REWEIGHT,
        n_particle = N_PARTICLE,
    ))

    function param(μ, σ)
        Params(
            α = 52.73019415710551,
            obs_sigma = 13.985483892291345,
            sample_cost = 0.0009382165229567387,
            switch_cost = 54.48621024645527,
            μ = μ,
            σ = σ
        )
    end

    function plogp(prm, particles=N_PARTICLE)
        smap(eachindex(data)) do i
            logp(prm, data[i], particles)
        end |> sum
    end


    space = Box(
        :µ => (0, 2 * μ_emp),
        :σ => (σ_emp / 4, 4 * σ_emp, :log),
    )

    g = 0:0.1:1
    @time G = asyncmap(Iterators.product(g, g); ntasks=10) do x
        print(".")
        prm = param(values(space(x))...)
        plogp(prm)
    end

    save(results, :prior_grid, G)
end
