using ClusterManagers, Sockets
using Distributed
using Distributions

myip = ip"10.2.159.72"
println("my ip is $myip")

cookie = length(ARGS) > 0 ? ARGS[1] : "cookie"

function smap(f, xs)
    pmap(f, xs;
        # on_error = e->(e isa ProcessExitedException ? NaN : rethrow()),
        on_error = e -> e,
        retry_delays = ExponentialBackOff(n = 3)
    )
end

function start_master()
    println("Creating ElasticManager")
        em = ElasticManager(
        addr=myip,
        port=58856,
        cookie=cookie,
        # topology=:master_worker
    )
    println("Waiting for workers..."); flush(stdout)
    while nprocs() == 1
        sleep(1)
    end
    sleep(5)
    println("Found ", nprocs(), " workers.")
    println("Testing parallelization.")
    smap(1:20) do i
        println(i); flush(stdout)
        sleep(0.01)
    end
end

function start_worker()
    elastic_worker(cookie, myip, 58856)
end