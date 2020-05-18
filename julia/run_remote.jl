using SplitApplyCombine

ips = split("""
3.84.13.98
54.163.166.197
52.90.75.243
54.157.7.54
34.227.161.30
107.22.132.251
54.172.14.3
3.83.2.207
54.210.114.122
35.175.141.74
54.165.182.110
52.91.94.217
54.210.231.194
3.92.188.248
""")

n_machine = length(ips)
max_job = 2500
RUN = "lesion19"

ssh_cmd(ip) = `ssh -i fred-ec2-key ubuntu@$ip`

function capture_stdout(dofunc, outfile)
    open(outfile, "w") do out
        redirect_stdout(out) do
            redirect_stderr(out) do
                dofunc()
            end
        end
    end
end

function add_known_host()
    println("add known host")
    asyncmap(ips) do ip
        ssh = ssh_cmd(ip)
        run(pipeline(`$ssh -o "StrictHostKeyChecking=no" "ls" `, devnull))
    end
end

function clear_results()
    asyncmap(ips) do ip
        ssh = ssh_cmd(ip)
        run(`$ssh 'rm -r results/lesion19'`)
    end
end

function run_quiet(cmd)
    run(pipeline(cmd, devnull))
end


function start(i; verbose=true)
    pp = verbose ? println : string
    rrun = verbose ? run : run_quiet
    ip = ips[i]
    # job = 1000*(i-1)+1:1000*i |> string
    job = i:n_machine:max_job |> string

    ssh = ssh_cmd(ip)

    pp("add known host")
    rrun(`$ssh -o "StrictHostKeyChecking=no" "ls" `)

    pp("push files")
    rrun(`$ssh "mkdir -p results"`)
    rrun(`rsync -rae "ssh -i fred-ec2-key" --exclude figs --exclude results --exclude scratch --exclude out ./ ubuntu@$ip:~/`)
    rrun(`$ssh 'rm -rf results/lesion19'`)

    # rrun(`rsync -rae "ssh -i fred-ec2-key" results/$RUN/ ubuntu@$ip:~/results/$RUN/`)
    # rrun(`$ssh "sudo apt install -y sysstat"`)

    # pp("install julia and precompile packages")
    # rrun(`$ssh 'bash setup.sh'`)

    pp("run job")
    rrun(`$ssh screen -d -m "./julia -p auto run_multi.jl both $job "`)
    println("started machine $ip")
end

function push_results()
    procs = map(ips) do ip
        run(`rsync -rae "ssh -i fred-ec2-key" --exclude *x results/$RUN/ ubuntu@$ip:~/results/$RUN/`; wait=false)
    end
    failed = findall(.!map(success, procs))
    if !isempty(failed)
        println("Failed to push: ", failed)
    end
end

function fetch_results()
    procs = map(ips) do ip
        run(`rsync -rae "ssh -i fred-ec2-key" ubuntu@$ip:~/results/$RUN/ results/$RUN/`; wait=false)
    end
    failed = findall(.!map(success, procs))
    if !isempty(failed)
        println("Failed to fetch: ", failed)
    end
end


function local_done(task="likelihood")
    x = read(`ls results/$RUN/$task`, String)
    sort(unique(parse.(Int, filter(x->!endswith(x, "x"), split(x)))))
end


function print_cpu_usage()
    xs = asyncmap(ips) do ip
        ssh = ssh_cmd(ip)
        try
            x = read(`$ssh "mpstat 3 1 | tail -1  | awk '{print \$3}'"`, String)
            # split(x, ".")[1]
            strip(x)
        catch
            "___"
        end
    end
    join(xs, "  ") |> println
end

function start_all()
    asyncmap(eachindex(ips)) do i
        try
            start(i, verbose=false)
        catch
            println("Could not start $(ips[i])")
        end
    end
end

function collect_done(task="likelihood")
    done = asyncmap([""; ips]) do ip
        if ip == ""
            x = read(`ls results/$RUN/$task`, String)
        else
            ssh = ssh_cmd(ip)
            x = read(`$ssh "ls results/$RUN/$task"`, String)
        end
        parse.(Int, filter(x->!endswith(x, "x"), split(x)))
    end |> flatten |> unique |> sort
end

function max_started(task="policies")
    asyncmap(ips) do ip
        ssh = ssh_cmd(ip)
        x = read(`$ssh "ls results/$RUN/$task"`, String)
        started = parse.(Int, filter(x->!endswith(x, "x"), split(x)))
        maximum(started)
    end |> maximum
end

function start_shutdown_watch()
    asyncmap(ips) do ip
        ssh = ssh_cmd(ip)
        run(`$ssh screen -d -m "sudo bash autoshutdown.sh"`)
    end
end






if false


while true
    fetch_results()
    println("Complete: ", length(local_done()))
    print_cpu_usage()
    sleep(60)
end


asyncmap(eachindex(ips)) do i
    try
        start(i, verbose=false)
    catch
        println("Could not start $(ips[i])")
    end
end





end


# # %% ====================  ====================
# done_ips = map(1:10) do i
#     ssh = ssh_cmd(ips[i])
#     println('*'^20, "  ", ips[i], "  ", '*'^20)
#     try
#         run(`$ssh "ls results/lesion_attention/likelihood/*x"`)
#         return missing
#     catch
#         return ips[i]
#     end
# end |> skipmissing |> collect

# # %% ====================  ====================
# asyncmap(1:10) do i
#     ip = ips[i]
#     ssh = ssh_cmd(ip)
#     run(`$ssh -o "StrictHostKeyChecking=no" "echo done" `)
#     read(`rsync -rae "ssh -i ~/.ssh/fred-ec2-key" ubuntu@$ip:~/results/ results/`)
# end

# # %% ====================  ====================







