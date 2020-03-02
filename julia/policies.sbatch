#!/usr/bin/env bash
#SBATCH --job-name=pol7500
#SBATCH --output=out/%A_%a
#SBATCH --array=1-2500
#SBATCH --time=180
#SBATCH --mem-per-cpu=500
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=flc2@princeton.edu

module load julia
julia compute_policies.jl `expr 7500 + $SLURM_ARRAY_TASK_ID`