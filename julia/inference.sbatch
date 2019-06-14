#!/usr/bin/env bash
#SBATCH --job-name=inf120
#SBATCH --output=out/%A_%a
#SBATCH --array=1-300
#SBATCH --time=120
#SBATCH --mem-per-cpu=3000
#SBATCH --cpus-per-task=1

module load julia
julia inference.jl worker
