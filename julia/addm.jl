using Random
using Distributions
include("human.jl")
include("utils.jl")

#=
f: fixated item
ft: a fixation time or a trial
v: a vector of item ratings/values
E: accumulated evidence
=#

@with_kw struct ADDM
    θ::Float64 = 0.3
    d::Float64 = .0002
    σ::Float64 = .02
end

# %% ==================== Binary Fixation Process ====================

abstract type FixationProcess end
abstract type Fixations end

struct BinaryFixationProcess <: FixationProcess
    prob_2_first::Float64
    first_durations::Dict{Int,Vector{Int}}
    other_durations::Dict{Int,Vector{Int}}
end
Base.show(io::IO, fp::BinaryFixationProcess) = print(io, "BinaryFixationProcess()")
difficulty(v) = abs(v[1] - v[2])

function BinaryFixationProcess(trials=load_dataset(2, :train))
    trials = filter(trials) do t
        difficulty(t.value) <= 5
    end
    prob_2_first = mean(first.(trials.fixations)) - 1
    first_durations = Dict(dif => Int[] for dif in unique(difficulty.(trials.value)))
    other_durations = Dict(dif => Int[] for dif in unique(difficulty.(trials.value)))
    for t in trials
        push!(first_durations[difficulty(t.value)], t.fix_times[1])
        push!(other_durations[difficulty(t.value)], t.fix_times[2:end-1]...)
    end
    BinaryFixationProcess(prob_2_first, first_durations, other_durations)
end

(fp::BinaryFixationProcess)(v) = BinaryFixations(fp, Int[], difficulty(v))

struct BinaryFixations <: Fixations
    fp::BinaryFixationProcess
    history::Vector{Int}
    dif::Int
end

function next!(fix::BinaryFixations)
    if isempty(fix.history)
        f = 1 + (rand() < fix.fp.prob_2_first)
        ft = rand(fix.fp.first_durations[fix.dif])
    else
        f = [2, 1][fix.history[end]]
        ft = rand(fix.fp.other_durations[fix.dif])
    end
    push!(fix.history, f)
    (f, ft)
end

# %% ==================== Binary Accumulation ====================

function simulate(m::ADDM, fp::BinaryFixationProcess, v::Vector{Int}; maxt=100000, save_history = false)
    @assert length(v) == 2
    fix = fp(v)
    noise = Normal(0, m.σ)
    history = Vector{Float64}[]
    E = 0.  # total accumulated evidence
    xx = m.d .* [v[1] - m.θ * v[2], m.θ * v[1] - v[2]]  # the two accumulation rates
    choice = 0
    ft = 0  # no initial fixationl
    fixations = Int[]
    fix_times = Int[]

    x = NaN  # have to initialize outside the if statement
    for t in 1:maxt
        t == maxt && error("Hit maximum time")
        ε = rand(noise)
        if ft == 0
            f, ft = next!(fix)
            push!(fixations, f); push!(fix_times, ft)
            x = xx[f]
        end
        E += x + ε
        ft -= 1
        save_history && push!(history, E)

        if !(-1 < E < 1)
            choice = E > 1 ? 1 : 2
            fix_times[end] -= ft  # remaining fixation time
            break
        end
    end
    (choice=choice, value=v, fixations=fixations, fix_times=fix_times), history
end

# # %% ==================== Trinary Fixation Process ====================
include("trinary_fixation_probs.jl")

@with_kw struct TrinaryFixationProcess <: FixationProcess
    # probabilities taken from KR 2011 SI. See trinary_fixation_probs.jl
    first_three::FixationProbs = parse_fixation_probs(_s3)
    all_seen::FixationProbs = parse_fixation_probs(_s4)
    some_unseen::FixationProbs = parse_fixation_probs(_s5)
    second_fix_rank_bonus::Float64 = .04
    third_fix_rank_bonus::Float64 = .03
    more_fix_rank_bonus::Float64 = .04
    durations::Dict{Tuple{Int64,Symbol},Vector{Int}} = fixation_durations(load_dataset(3, :full))
end
Base.show(io::IO, fp::TrinaryFixationProcess) = print(io, "TrinaryFixationProcess()")

# Fixation times were randomly sampled directly from the vector of measured
# nonfinal fixations, conditional on the value of the fixated item, and whether
# it was a first, second, or other fixation.

fixtype(fixnum) = [:first, :second, :other][min(fixnum, 3)]

function fixation_durations(trials)
    durations = Dict((value, kind) => Int[] 
        for value in trials.value |> flatten |> unique, 
            kind in [:first, :second, :other])

    for t in trials
        for i in eachindex(t.fixations)[1:end-1]
            val = t.value[t.fixations[i]]
            push!(durations[val, fixtype(i)], t.fix_times[i])
        end
    end
    durations
end

struct TrinaryFixations <: Fixations
    fp::TrinaryFixationProcess
    history::Vector{Int}
    value::Vector{Int}
    rank::Vector{Int}
    unrank::Vector{Int}
    seen::Set{Int}
end

(fp::TrinaryFixationProcess)(v) = TrinaryFixations(fp, Int[], v, sortperm(sortperm(v)), sortperm(v), Set{Int}())

function next!(fix::TrinaryFixations)
    fixnum = length(fix.history) + 1
    f = if fixnum <= 3
        rand(fix.fp.first_three[fix.history])
    else
        last_two = fix.history[end-1:end]
        if length(fix.seen) == 3
            fixrank = rand(fix.fp.all_seen[fix.rank[last_two]])
            fix.unrank[fixrank]
        else
            rand(fix.fp.some_unseen[last_two])
        end
    end
    ft = rand(fix.fp.durations[fix.value[f], fixtype(fixnum)])
    push!(fix.history, f); push!(fix.seen, f)
    (f, ft)
end


# %% ==================== Trinary Simulation ====================


function downweight!(v, f, θ)
    for i in eachindex(v)
        if i != f
            v[i] *= θ
        end
    end
    v
end

function check_termination(E)
    E[1] - max(E[2], E[3]) >= 1 && return 1
    E[2] - max(E[1], E[3]) >= 1 && return 2
    E[3] - max(E[1], E[2]) >= 1 && return 3
    return 0
end

function simulate(m::ADDM, fp::TrinaryFixationProcess, v::Vector{Int}; maxt=100000, save_history = false)
    fix = fp(v)
    N = length(v); @assert N == 3
    noise = Normal(0, m.σ / √2)  # see Krajbich 2011 SI results
    history = Vector{Float64}[]
    E = zeros(N)  # total accumulated evidence
    x = zeros(N)  # momentary evidence
    ε = zeros(N)  # momentary noise
    choice = 0
    ft = 0  # no initiaal fixation
    fixations = Int[]
    fix_times = Int[]

    for t in 1:maxt
        rand!(noise, ε)  # resample noise (in place)
        if ft == 0
            f, ft = next!(fix)
            push!(fixations, f); push!(fix_times, ft)
            x = downweight!(m.d .* v, f, m.θ)
        end
        E .+= x .+ ε
        ft -= 1
        save_history && push!(history, copy(E))

        choice = check_termination(E)
        if choice != 0
            fix_times[end] -= ft  # remaining fixation time
            break
        end
    end
    (choice=choice, value=v, fixations=fixations, fix_times=fix_times), history
end

function simulate_trials(m::ADDM, trials::Table)
    n_item = length(trials.value[1])
    fp = Dict(2 => BinaryFixationProcess, 3 => TrinaryFixationProcess)[n_item]()
    map(trials.value) do v
        simulate(m, fp, v)[1]
    end |> Table
end

