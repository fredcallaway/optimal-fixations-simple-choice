using ClusterManagers, Sockets
using Distributed

myip = ip"10.2.159.72"
# myip = ip"10.36.16.11"
# myip = getipaddr()

# println("my ip is $myip")

# cookie = length(ARGS) > 0 ? ARGS[1] : "cookie"
cookie = "cookie"

function smap(f, xs)
    pmap(f, xs;
        # on_error = e->(e isa ProcessExitedException ? NaN : rethrow()),
        on_error = e -> (println(e), throw(e)),
        retry_delays = ExponentialBackOff(n = 3)
    )
end

function start_master(;wait=true, test=false)
    println("Creating ElasticManager")
    em = ElasticManager(
        addr=myip,
        port=58856,
        cookie=cookie,
        topology=:master_worker
    )
    if wait
        println("Waiting for workers..."); flush(stdout)
        while nprocs() == 1
            sleep(1)
        end
        sleep(15)
    end
    println("Found ", nprocs(), " workers.")
    if test
        println("Testing parallelization.")
        smap(1:20) do i
            println(i); flush(stdout)
            sleep(0.01)
        end
    end
    return em
end

function start_worker()
    println("Beginning work for: ", cookie)
    elastic_worker(cookie, myip, 58856)
end