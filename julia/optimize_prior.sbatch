#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --output=out/%A_%a
#SBATCH --array=1-100
#SBATCH --time=59
#SBATCH --mem-per-cpu=5000
#SBATCH --cpus-per-task=1

module load julia
julia optimize_prior.jl rando $SLURM_ARRAY_TASK_ID
