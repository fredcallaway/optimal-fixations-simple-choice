ips = split("""
35.170.187.128
3.82.195.110
184.72.100.250
54.85.178.111
3.90.53.185
""")

n_machine = length(ips)

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

function fetch_results()
    asyncmap(ips) do ip
        ssh = ssh_cmd(ip)
        run(`rsync -rae "ssh -i fred-ec2-key" ubuntu@$ip:~/results/ results/`)
    end
end

    


function start()
    for (i, ip) in enumerate(ips)
        # job = 1000*(i-1)+1:1000*i |> string
        job = i:n_machine:10000 |> string

        ssh = ssh_cmd(ip)
        println(ssh)

        println("add known host")
        run(`$ssh -o "StrictHostKeyChecking=no" "ls" `)

        println("push files")
        run(`rsync -rae "ssh -i /Users/fred/.ssh/fred-ec2-key" --exclude figs --exclude results --exclude scratch ./ ubuntu@$ip:~/`)

        println("install julia and precompile packages")
        # run(pipeline(`$ssh "sudo apt install -y sysstat"`, devnull))
        # run(`$ssh 'bash setup.sh &> /dev/null'`)
        run(`$ssh './julia install_deps.jl'`)

        println("run job")
        run(`$ssh screen -d -m "./julia -p auto run_multi.jl both $job"`)
    end
end



# %% ====================  ====================

function print_cpu_usage()
    xs = asyncmap(ips) do ip
        ssh = ssh_cmd(ip)
        x = read(`$ssh "mpstat 1 1 | tail -1  | awk '{print \$3}'"`, String)
        # split(x, ".")[1]
        x
    end
    join(xs, "  ") |> println
end

print_cpu_usage()


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







