include("model_base.jl")
include("new_moment_base.jl")

FOLD = "odd"
N_TOP = 80
μ_grain = 0.1

mkpath("results/moment_ucb")
mkpath("results/moment_ucb_extra")

function compute_moment_loss(job)
    dest = "results/moment_ucb/$job"
    if isfile(dest)
        println("Job $job has been completed")
        return
    elseif isfile(dest * "x")
        println("Job $job is in progress")
        return
    end
    touch(dest*"x")
    try
        println("Computing moment loss for job ", job)
        
        datasets = map(["two", "three"]) do num
            build_dataset(num; fold=FOLD)
        end

        descriptors = [choice_value, n_fix, total_fix_time, chosen_relative_fix]

        ucb = deserialize("results/preopt_ucb/$job");

        results = map(0:μ_grain:1) do β_μ
            losses = map(1:2) do item_idx
                policies, μ, sem = ucb[item_idx]
                ds = datasets[item_idx]
                loss_fs = make_loss.(descriptors, [ds])
                top = policies[partialsortperm(-μ, 1:N_TOP)]
                sim = simulate(top, ds, β_μ)
                losses = [f(sim) for f in loss_fs]
                # serialize("results/moment_ucb_extra/sim-$job-$β_μ-$item_idx", sim)
                # serialize("results/moment_ucb_extra/losess-$job-$β_μ-$item_idx", losses)
                sum(losses)
            end
            pol = ucb[1][1][1];
            prm = (namedtuple(type2dict(pol.m))..., α=pol.α, β_μ=β_μ)
            prm, losses
        end

        serialize(dest, results)
        println("Wrote $dest")
    finally
        rm(dest*"x")
    end
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    job = parse(Int, ARGS[1])
    compute_moment_loss(job)
end