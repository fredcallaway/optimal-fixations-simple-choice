using Distributed
# addprocs(["griffiths-gpu02.pni.princeton.edu"], topology=:master_worker)
@everywhere begin
    cd("/usr/people/flc2/juke/choice-eye-tracking/julia/")
    include("model.jl")
    include("job.jl")
    include("human.jl")
    include("simulations.jl")
    include("loss.jl")
    CUTOFF = 2000

    function load_policy(job)
        m = MetaMDP(job)
        try
            Policy(m, deserialize(job, :optim).θ1)
        catch
            missing
        end
    end

    _loss_funcs = [
        make_loss(value_choice),
        make_loss(fixation_bias),
        make_loss(value_bias),
        make_loss(fourth_rank, :integer),
        make_loss(first_fixation_duration),
        make_loss(last_fixation_duration),
        make_loss(difference_time),
        make_loss(difference_nfix),
        make_loss(fixation_times, :integer),
        make_loss(last_fix_bias),
        make_loss(gaze_cascade, :integer),
        make_loss(fixate_on_best, Binning(0:CUTOFF/7:CUTOFF)),
        # make_loss(old_value_choice, :integer),
        # make_loss(fixation_value, Binning(0:3970/20:3970)),
    ]
    loss(sim) = sum(ℓ(sim) for ℓ in _loss_funcs)
    breakdown_loss(sim) = [ℓ(sim) for ℓ in _loss_funcs]
    µs = 0:0.1:µ_emp
end
# %% ====================  ====================

using Glob
files = glob("runs/rando1000/jobs/*")
jobs = Job.(files)

job = jobs[4]

errors = pmap(enumerate(jobs)) do (i, job)
    try
        pol = load_policy(job)
        ismissing(pol) && return 0
        exists(job, :simulations) && return 0
        x = map(μs) do μ
            # (prior=(μ, σ_emp), sim=nothing, loss=rand())
            sim = simulate_experiment(pol, (µ, σ_emp))
            (prior=(μ, σ_emp), sim=sim, losses=breakdown_loss(sim))
        end
        serialize(job, :simulations, x)
    catch
        println("Error ", i)
        return 1
    end
    return 0
end
println(sum(errors), " errors")
