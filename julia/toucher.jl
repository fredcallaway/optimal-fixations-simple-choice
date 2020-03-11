struct Toucher
    file::String
    interval::Real
    done::Channel{Bool}
end

Toucher(file, interval=60) = Toucher(file, interval, Channel{Bool}(1))

function run!(t::Toucher)
    isready(t.done) && take!(t.done)
    open(t.file, "w") do f
        write(f, Libc.gethostname())
    end
    @async while !isready(t.done)
        touch(t.file)
        sleep(t.interval)
    end
    return t
end

function stop!(t::Toucher)
    put!(t.done, true)
    try
        rm(t.file)
    catch end
end

isactive(t; tolerance=0) = (time() - mtime(t.file)) < (tolerance + t.interval)
