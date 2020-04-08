quick:
	# n_arm obs_sigma sample_cost switch_cost n_iter n_roll n_sim seed
	julia -L "optimize.jl" -e "main(3, 5, 0.005, 0.005, 4, 4, 4, 0, \"quick\")"

f1:
	rsync -av --exclude .git --exclude figs --exclude tex --exclude results ./ f1:~/cet/
f2:
	rsync -av --exclude .git --exclude figs --exclude tex --exclude results ./ f2:~/cet/
f3:
	rsync -av --exclude .git --exclude figs --exclude tex --exclude results ./ f3:~/cet/
# 	fswatch -o . | while read f
# 	do
# 		rsync -av --exclude .git --exclude figs --exclude tmp ./ f1:~/cet/
# 	done