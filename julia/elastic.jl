using ClusterManagers, Sockets
using Distributed

const COOKIE = "cookie"
# const MY_IP = ip"10.0.20.16"  # scotty

const MY_IP = ip"10.2.159.72"
const PORT = 58857

# const MY_IP = ip"128.112.221.197"
# const PORT = 58005

function smap(f, xs)
    pmap(f, xs;
        # on_error = e->(e isa ProcessExitedException ? NaN : rethrow()),
        # on_error = e -> (println(e), throw(e)),
        retry_delays = ExponentialBackOff(n = 3)
    )
end

function start_master(cookie=COOKIE; wait=15, test=false)
    println("Creating ElasticManager")
    em = ElasticManager(
        addr=MY_IP,
        port=PORT,
        cookie=COOKIE,
        topology=:master_worker
    )
    if wait > 0
        println("Waiting for workers..."); flush(stdout)
        while nprocs() == 1
            sleep(1)
        end
        sleep(wait)
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

function start_worker(cookie=COOKIE)
    println("Beginning work for: ", COOKIE)
    elastic_worker(COOKIE, MY_IP, PORT)
end