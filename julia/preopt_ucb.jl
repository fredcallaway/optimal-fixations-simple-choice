out = "results/grid/ucb"
mkpath(out)


using Distributed
using Serialization
include("box.jl")
include("meta_mdp.jl")
include("bmps.jl")
include("optimize_bmps.jl")
include("ucb_bmps.jl")

space = Box(
    :sample_time => 100,
    # :α => (50, 200),
    :α => 200,
    :σ_obs => (1, 5),
    :sample_cost => (.002, .006),
    :switch_cost => (.01, .05),
)

function get_prm(job)
    # seq = SobolSeq(n_free(space))
    # skip(seq, job-1; exact=true)
    # x = next!(seq)
    g = range(0,1,length=11)
    mg = Iterators.product(repeat([g], n_free(space))...)
    x = collect(collect(mg)[job])
    x |> space |> namedtuple
end

function run_ucb(job)
    dest = "$out/$job"
    if isfile(dest)
        println("Job $job has been completed")
        return
    elseif isfile(dest * "x")
        println("Job $job is in progress")
        return
    end
    touch(dest*"x")
    try
        println("Running UCB for job ", job)
        flush(stdout)
        prm = get_prm(job)

        results = map(2:3) do n_item
            m = MetaMDP(n_item, prm.σ_obs, prm.sample_cost, prm.switch_cost)
            policies, μ, sem = ucb(m; N=8000, α=prm.α, n_iter=500, n_init=100, n_roll=100, n_top=80)
            policies, μ, sem
        end

        serialize(dest, results)
        println("Wrote $dest")
    finally
        rm(dest*"x")
    end
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    job = parse(Int, ARGS[1])
    run_ucb(job)
end
