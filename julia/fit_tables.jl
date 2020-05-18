include("utils.jl")
using SplitApplyCombine
using Serialization
using Statistics
using DataStructures: OrderedDict

sprintf(fmt::String,args...) = @eval @sprintf($fmt,$(args...))
fmtfloat(x, d) = sprintf("%.$(d)f", x)

function mean_std_str(xs, d)
    fmt = "%.$(d)f ± %.$(d)f"
    sprintf(fmt, juxt(mean, std)(xs)...)
end

# %% ====================  ====================

pnames = Symbol.(split("σ_obs sample_cost switch_cost α p_switch p_stop test_like"))
digs = [2, 4, 3, 0, 3, 3, 0]

models = OrderedDict(
    "main14" => "Full Model",
    "lesion19" => "Random Fixations",
    # "rando17" => "Random Fixations and Stopping",
)

function load_result(r)
    prms = deserialize("results/$r/best_parameters/joint-false")
    test_like = map(eachindex(prms)) do i
        x = deserialize("results/$r/test_likelihood/joint-false/$i")
        x[1][1] + x[2][1]
    end
    (invert(prms)..., test_like=test_like)
end

# %% ====================  ====================
mkpath("results/comp")
open("results/comp/table.tex", "w") do f
    function w(x)
        print(x)
        write(f, x)
    end
    for r in keys(models)
        w(models[r])
        w(" & ")
        P = load_result(r)
        foreach(pnames, digs) do pn, d
            if hasfield(typeof(P), pn) && !all(getfield(P, pn) .≈ 0) && !all(isnan.(getfield(P, pn)))
                w(mean_std_str(getfield(P, pn), d))
            end
            w(pn == :test_like ? " \\\\\n" : " & ")
        end
    end
end






# # %% ====================  ====================

# let
#     parameters = [:σ_obs, :sample_cost, :switch_cost, :β_μ, :α]
#     rounding_digits = [2, 4, 3, 2, 1]
#     print("Fit mode   Fit prior     ")
#     for k in parameters
#         @printf "%-17s" k
#     end
#     println()
#     for num in ["two", "three", "joint"]
#         for fit_prior in [false, true]
#             prms = invert(deserialize("$BASE_DIR/best_parameters/$num-$fit_prior"))
#             @printf "%8s  %10s  " num fit_prior
#             for (k, r) in zip(parameters, rounding_digits)
#                 mx, sx = juxt(mean, std)(getfield(prms, k))
#                 @printf "%7s ± %-7s"  round(mx; digits=r) round(sx; digits=r+1)
#                 # print(round(mx; digits=r), " ± ", round(sx; digits=r+1), "   ")
#             end
#             println()
#         end
#     end
# end


# # %% ====================  ====================

# let
#     # @printf "%10s  %10s  " "Fit mode" "Fit prior\n"
#     println("Fit mode   Fit prior  Two items            Three items")
#     for fit_mode in ["joint", "separate"]
#         for fit_prior in [false, true]
#             !isdir("$BASE_DIR/test_likelihood/$fit_mode-$fit_prior/") && continue
#             l2, l3 = map(1:30) do i
#                 map(first, deserialize("$BASE_DIR/test_likelihood/$fit_mode-$fit_prior/$i"))
#             end |> invert
#             @printf "%8s  %10s  " fit_mode fit_prior
#             println(mean_std_str(l2), "   ", mean_std_str(l3))
#         end
#     end
# end


