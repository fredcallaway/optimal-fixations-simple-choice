include("bern2.jl")
display("")
s  = ([(10,4), (10,10)], 1)
Q(s, 1)
@time V(INITIAL);

# _V[INITIAL]
# %% ====================  ====================
@code_warntype Q(INITIAL, 1)
@code_warntype V(INITIAL)
# %% ====================  ====================
State_

println(1)
using Profile
s = ([(10,10), (10,10)], 1)
Profile.clear()
@profile Q(INITIAL, 1);
Profile.print()
