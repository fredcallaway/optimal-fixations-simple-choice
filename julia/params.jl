
@with_kw mutable struct Params
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

MetaMDP(n_arm, prm::Params) = MetaMDP(
    n_arm,
    prm.σ_obs,
    prm.sample_cost,
    prm.switch_cost,
)