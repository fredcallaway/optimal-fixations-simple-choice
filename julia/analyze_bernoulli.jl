using Serialization
using Glob
using TypedTables
using Query

include("bernoulli_metabandits.jl")

res = open.(deserialize, glob("bernoulli/*"))

tbl = map(res) do r
    (
        bmps_val = r.bmps_v,
        opt_val = r.opt_val,
        bmps_steps = r.bmps_steps,
        opt_steps = r.opt_steps,
        switch_cost = r.m.switch_cost,
        sample_cost = r.m. sample_cost,
    )
end |> Table

tbl |>
    @groupby(_.switch_cost) |>
    @map({x=key(_), loss=_.bmps_val - _.opt_val}) |>
    Table