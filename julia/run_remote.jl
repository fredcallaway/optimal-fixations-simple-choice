ips = split("""
52.202.36.6
34.239.130.176
3.84.220.166
3.93.231.208
52.1.68.94
3.88.57.33
54.146.132.222
3.86.38.37
54.226.176.113
54.175.12.69
""")

ssh_cmd(ip) = `ssh -i ~/.ssh/fred-ec2-key ubuntu@$ip`

function start()
    for i in 3:10
        job = 1000*(i-1)+1:1000*i |> string
        ip = ips[i]

        ssh = ssh_cmd(ip)
        println(ssh)

        println("add known host")
        run(`$ssh -o "StrictHostKeyChecking=no" "ls" `)

        run(pipeline(`$ssh "sudo apt install -y sysstat"`, devnull))

        println("push files")
        run(`rsync -rae "ssh -i /Users/fred/.ssh/fred-ec2-key" --exclude figs --exclude results --exclude scratch ./ ubuntu@$ip:~/`)

        println("install julia and precompile packages")
        run(`$ssh 'bash setup.sh &> /dev/null'`)

        println("run job")
        run(`$ssh screen -d -m "./julia -p auto run_multi.jl both $job"`)
    end
end

# %% ====================  ====================

function print_cpu_usage()
    xs = asyncmap(1:10) do i
        ssh = ssh_cmd(ip)
        x = read(`$ssh "mpstat 1 1 | tail -1  | awk '{print \$3}'"`, String)
        # split(x, ".")[1]
        x
    end
    join(xs, "  ") |> println
end


# %% ====================  ====================
done_ips = map(1:10) do i
    ssh = ssh_cmd(ips[i])
    println('*'^20, "  ", ips[i], "  ", '*'^20)
    try
        run(`$ssh "ls results/lesion_attention/likelihood/*x"`)
        return missing
    catch
        return ips[i]
    end
end |> skipmissing |> collect

# %% ====================  ====================
asyncmap(1:10) do i
    ip = ips[i]
    ssh = ssh_cmd(ip)
    run(`$ssh -o "StrictHostKeyChecking=no" "echo done" `)
    read(`rsync -rae "ssh -i ~/.ssh/fred-ec2-key" ubuntu@$ip:~/results/ results/`)
end

# %% ====================  ====================







