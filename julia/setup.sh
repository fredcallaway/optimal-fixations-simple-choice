if [ ! -f ./julia ]; then
    echo "Install julia"
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.1-linux-x86_64.tar.gz
    tar -xf julia-1.3.1-linux-x86_64.tar.gz
    ln -s /home/ubuntu/julia-1.3.1/bin/julia julia
fi
if ! [ -x "$(command -v mpstat)" ]; then
    echo "Install sysstat"
    sudo apt install -y sysstat
fi
./julia install_deps.jl
bash setup_efs.sh