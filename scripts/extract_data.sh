#!/bin/bash

#SBATCH --job-name=extract_data
#SBATCH --partition=standard
#SBATCH --time=00-10:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --export=ALL
#SBATCH --output=./logs/%x-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
echo "Starting job script..."

module load python/3.11.5

srun pdm run python extract_data.py

echo "Finishing job script..."