#!/usr/bin/env bash
#SBATCH --job-name=policies
#SBATCH --output=out/%A_%a
#SBATCH --array=1-1000
#SBATCH --time=240
#SBATCH --mem-per-cpu=500
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=flc2@princeton.edu

module load julia
julia compute_policies.jl $SLURM_ARRAY_TASK_ID