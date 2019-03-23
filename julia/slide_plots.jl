
# %% ====================  ====================
using StatPlots
function plotnorm!(μ, σ, label="")
    d = Normal(μ, σ)
    plot!(d, label="", line=(:black, 2), ticks=false)
    p = pdf(d, μ)
    annotate!([(μ, p+0.05, text(label, 32))])
end


# %% ====================  ====================
plot()
x = 2.5
plotnorm!(0,1, "A")
plotnorm!(1x,1, "B")
plotnorm!(2x,1, "C")
ylims!(0, 0.5)
savefig("figs/example.pdf")

# %% ====================  ====================
plot()
plotnorm!(0,1, "A")
plotnorm!(3,1, "B")
plotnorm!(4,1, "C")
ylims!(0, 0.5)
savefig("figs/example1.pdf")

# %% ====================  ====================
plot()
plotnorm!(0,1, "A")
plotnorm!(1,1, "B")
plotnorm!(4,1, "C")
ylims!(0, 0.5)
savefig("figs/example2.pdf")
