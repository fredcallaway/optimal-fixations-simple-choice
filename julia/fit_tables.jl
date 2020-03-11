include("fit_base.jl")
using SplitApplyCombine

mean_std_str(xs) = @sprintf("%.3f ± %.3f", juxt(mean, std)(xs)...)

# %% ====================  ====================

let
    parameters = [:σ_obs, :sample_cost, :switch_cost, :β_μ, :α]
    rounding_digits = [2, 4, 3, 2, 1]
    print("Fit mode   Fit prior     ")
    for k in parameters
        @printf "%-17s" k
    end
    println()
    for num in ["two", "three", "joint"]
        for fit_prior in [false, true]
            prms = invert(deserialize("$BASE_DIR/best_parameters/$num-$fit_prior"))
            @printf "%8s  %10s  " num fit_prior
            for (k, r) in zip(parameters, rounding_digits)
                mx, sx = juxt(mean, std)(getfield(prms, k))
                @printf "%7s ± %-7s"  round(mx; digits=r) round(sx; digits=r+1)
                # print(round(mx; digits=r), " ± ", round(sx; digits=r+1), "   ")
            end
            println()
        end
    end
end


# %% ====================  ====================

let
    # @printf "%10s  %10s  " "Fit mode" "Fit prior\n"
    println("Fit mode   Fit prior  Two items            Three items")
    for fit_mode in ["joint", "separate"]
        for fit_prior in [false, true]
            !isdir("$BASE_DIR/test_likelihood/$fit_mode-$fit_prior/") && continue
            l2, l3 = map(1:30) do i
                map(first, deserialize("$BASE_DIR/test_likelihood/$fit_mode-$fit_prior/$i"))
            end |> invert
            @printf "%8s  %10s  " fit_mode fit_prior
            println(mean_std_str(l2), "   ", mean_std_str(l3))
        end
    end
end


