"""
    hyperband(getconfig, getloss, maxresource=27, reduction=3)
Hyperparameter optimization using the hyperband algorithm from ([Lisha
et al. 2016](https://arxiv.org/abs/1603.06560)).  You can try a simple
MNIST example using `Knet.hyperband_demo()`. 
# Arguments
* `getconfig()` returns random configurations with a user defined type and distribution.
* `getloss(c,n)` returns loss for configuration `c` and number of resources (e.g. epochs) `n`.
* `maxresource` is the maximum number of resources any one configuration should be given.
* `reduction` is an algorithm parameter (see paper), 3 is a good value.
"""
function hyperband(getconfig, getloss; maxresource=27, reduction=3)
    smax = floor(Int, log(maxresource)/log(reduction))
    B = (smax + 1) * maxresource
    best = (Inf,)
    for s in smax:-1:0
        n = ceil(Int, (B/maxresource)*((reduction^s)/(s+1)))
        r = maxresource / (reduction^s)
        curr = halving(getconfig, getloss, n, r, reduction, s)
        if curr[1] < best[1]; (best=curr); end
    end
    return best
end

# TODO: document Successive/Sequential Halving:
# http://www.jmlr.org/proceedings/papers/v51/jamieson16.pdf,
# http://www.jmlr.org/proceedings/papers/v28/karnin13.pdf,
# Successive  Reject:
# http://certis.enpc.fr/~audibert/Mes%20articles/COLT10.pdf
function halving(getconfig, getloss, n; r=1, reduction=3, s=round(Int, log(n)/log(reduction)))
    best = (Inf,)
    T = [ getconfig() for i=1:n ]
    for i in 0:s
        ni = floor(Int,n/(reduction^i))
        ri = r*(reduction^i)
        # println((:s,s,:n,n,:r,r,:i,i,:ni,ni,:ri,ri,:T,length(T)))
        L = [ getloss(t, ri) for t in T ]
        l = sortperm(L); l1=l[1]
        best = (L[l1],ri,T[l1])
        # L[l1] < best[1] && (best = (L[l1],ri,T[l1])) # ;println("best1: $best"))
        T = T[l[1:floor(Int,ni/reduction)]]
        println(best)
    end
    # println("best2: $best")
    return best
end