
# %% ==================== Meta Greedy ====================

function voc1(m::MetaMDP, b::Belief, c::Int)
    c == TERM && return 0.
    q = mapreduce(+, results(m, b, c)) do (p, b1, r)
        p * (term_reward(m, b1) + r)
    end
    q - term_reward(m, b)
end

voc1(m, b) = [voc1(m, b, c) for c in 0:length(b)]

function meta_greedy(cost)
    t -> begin
        m = make_mdp(t.graph, cost)
        map(t.bs) do b
            voc1(m, b)
        end
    end
end

fit_error_model(map(meta_greedy(1.0), trials), trials)


# %% ==================== Best First ====================

function best_first_value(m, b, sat_thresh, prune_thresh)
    # FIXME: how do we handle observing the start or end cities?
    nv = fill(-1e10, length(m))
    for p in paths(m)
        v = path_value(m, b, p)
        for i in p
            nv[i] = max(nv[i], v)
        end
    end
    for i in eachindex(nv)
        if !isnan(b[i])  # already observed
            nv[i] = -Inf
        elseif nv[i] < prune_thresh
            nv[i] = -1e10
        end
    end
    term_value = (maximum(nv) >= sat_thresh ? Inf : -1e5)
    [term_value; nv]
end



function best_first(sat_thresh, prune_thresh)
    t -> begin
        m = make_mdp(t.graph, NaN)
        map(t.bs) do b
            best_first_value(m, b, sat_thresh, prune_thresh)
        end
    end
end


fit_error_model(map(best_first(Inf, -Inf), trials), trials; x0=[0.01, 0.99])

fit_error_model(map(best_first(-100, -300), trials), trials; x0=[0.01, 0.99])


# %% ====================  ====================

qs = map(best_first(-200, -100), trials)
logp(qs, trials, 5., 1.)


t = trials[1]
best_first_value(m, t.bs[1], -200, -100)





