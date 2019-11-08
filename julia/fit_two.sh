# mkdir out/ind1/
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 10 --fold 1/10 &> out/1 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 11 --fold 1/10 &> out/2 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 13 --fold 1/10 &> out/3 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 14 --fold 1/10 &> out/4 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 16 --fold 1/10 &> out/5 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 17 --fold 1/10 &> out/6 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 18 --fold 1/10 &> out/7 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 19 --fold 1/10 &> out/8 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 20 --fold 1/10 &> out/9 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 22 --fold 1/10 &> out/10 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 23 --fold 1/10 &> out/11 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 25 --fold 1/10 &> out/12 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 26 --fold 1/10 &> out/13 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 27 --fold 1/10 &> out/14 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 28 --fold 1/10 &> out/15 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 29 --fold 1/10 &> out/16 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 30 --fold 1/10 &> out/17 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 31 --fold 1/10 &> out/18 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 32 --fold 1/10 &> out/19 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 33 --fold 1/10 &> out/20 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 34 --fold 1/10 &> out/21 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 35 --fold 1/10 &> out/22 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 38 --fold 1/10 &> out/23 &
julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 39 --fold 1/10 &> out/24 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 40 --fold 1/10 &> out/25 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 41 --fold 1/10 &> out/26 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 42 --fold 1/10 &> out/27 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 44 --fold 1/10 &> out/28 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 45 --fold 1/10 &> out/29 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 46 --fold 1/10 &> out/30 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 47 --fold 1/10 &> out/31 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 48 --fold 1/10 &> out/32 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 49 --fold 1/10 &> out/33 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 51 --fold 1/10 &> out/34 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 52 --fold 1/10 &> out/35 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 53 --fold 1/10 &> out/36 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 54 --fold 1/10 &> out/37 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 55 --fold 1/10 &> out/38 &
nice -n 12 julia -p 4 fit_pseudo.jl --res ind1 --dataset two --subject 56 --fold 1/10 &> out/39 &

echo "And now, we wait..."
wait
