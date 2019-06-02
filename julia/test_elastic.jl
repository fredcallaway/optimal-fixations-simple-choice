include("elastic.jl")


function work(i)
    println("I am doing job $i")
    sleep(0.5)
end

if get(ARGS, 1, "") == "worker"
    start_worker()
else
    start_master()
    results = smap(work, 1:50)
    # @distributed (+) for i in 1:100
    #     sleep(1)
    #     1
    # end
end
