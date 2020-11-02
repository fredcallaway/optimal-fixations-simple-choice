using DataStructures: DefaultDict, CircularBuffer
_s3 = """
1 => 37
1 2 => 29
1 2 3 => 90
1 2 1 => 10
1 3 => 71
1 3 2 => 85
1 3 1 => 15
2 => 46
2 1 => 60
2 1 3 => 89
2 1 2 => 11 
2 3 => 40
2 3 1 => 90
2 3 2 => 10
3 => 17
3 1 => 67
3 1 2 => 81
3 1 3 => 19
3 2 => 33
3 2 1 => 85
3 2 3 => 15
"""

_s4 = """
3 2 3 => 57
3 2 1 => 43
3 1 2 => 57
3 1 3 => 43
2 3 1 => 55
2 3 2 => 45
2 1 3 => 69
2 1 2 => 31
1 3 2 => 79
1 3 1 => 21
1 2 3 => 76
1 2 1 => 24
"""

_s5 = """
1 2 3 => 75
1 2 1 => 25
1 3 2 => 72
1 3 1 => 28
2 1 3 => 82
2 1 2 => 18
2 3 1 => 87
2 3 2 => 13
3 1 2 => 70
3 1 3 => 30
3 2 1 => 84
3 2 3 => 16
"""
FixationProbs = Dict{Array{Int64,1},DiscreteNonParametric{Int64,Float64,Array{Int64,1},Array{Float64,1}}}

function parse_fixation_probs(datastring)::FixationProbs
    lines = map(split(strip(datastring), "\n")) do line
        f, p = split(line, " => ")
        parse.(Int, split(f, " ")), parse(Int, p) / 100
    end

    probs = DefaultDict{Vector{Int},Dict{Int,Float64}}(Dict)
    for (seq, p) in lines
        probs[seq[1:end-1]][seq[end]] = p
    end

    valmap(probs) do d
        DiscreteNonParametric(collect(keys(d)), collect(values(d)))
    end
end

