#!/usr/bin/env bash
#SBATCH --job-name=run_simulations
#SBATCH --output=out/%A_%a
#SBATCH --array=1-300
#SBATCH --time=120
#SBATCH --mem-per-cpu=1000
#SBATCH --cpus-per-task=1

module load julia
julia run_simulations.jl
