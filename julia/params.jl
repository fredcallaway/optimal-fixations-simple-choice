
@with_kw mutable struct Params
    n_arm::Int
    α::Float64
    σ_obs::Float64
    sample_cost::Float64
    switch_cost::Float64
    µ::Float64
    σ::Float64
    sample_time::Float64
    σ_rating::Float64
end
Params(d::AbstractDict) = Params(;d...)

MetaMDP(prm::Params) = MetaMDP(
    prm.n_arm,
    prm.σ_obs,
    prm.sample_cost,
    prm.switch_cost,
)