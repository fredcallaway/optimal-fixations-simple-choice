wget https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.1-linux-x86_64.tar.gz
tar -xf julia-1.3.1-linux-x86_64.tar.gz
ln -s /home/ubuntu/julia-1.3.1/bin/julia julia
./julia install_deps.jl