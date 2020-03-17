using Sobol
using Serialization

include("box.jl")
include("meta_mdp.jl")
include("bmps.jl")
include("toucher.jl")


# include("runs/sobol4.jl")
# include("runs/lesion.jl")
# include("runs/no_switch_cost.jl")
include("runs/attention_noise.jl")

mkpath(BASE_DIR)

function get_prm(job)
    x = if SEARCH_STRATEGY == :sobol
        seq = SobolSeq(n_free(SPACE))
        skip(seq, job-1; exact=true)
        next!(seq)
    elseif SEARCH_STRATEGY == :grid
        g = range(0,1,length=GRID_SIZE)
        mg = Iterators.product(repeat([g], n_free(SPACE))...)
        collect(collect(mg)[job])
    else
        error("Invalid search strategy: $SEARCH_STRATEGY")
    end

    x |> SPACE |> namedtuple
end



function do_job(f::Function, name::String, job::Int; force=false)
    out = "$BASE_DIR/$name"
    mkpath(out)
    dest = "$out/$job"
    if isfile(dest)
        print("$dest already exists, ")
        if force
            println("overwriting")
        else
            println("skipping")
            return
        end
    end
    
    t = Toucher(dest * "x")
    if isactive(t; tolerance=600)
        print("$dest is currently in progress, ")
        if force
            println("overwriting")
        else
            println("skipping")
            return
        end
    end
    run!(t)
    
    try
        println("Computing $dest"); flush(stdout)
        @time results = f(job)
        serialize(dest, results)
        println("Wrote $dest"); flush(stdout)
    finally
        stop!(t)
    end
end




