#!/bin/bash
#SBATCH -A 65015016
#SBATCH --job-name=65015016
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -o out_%j.txt
#SBATCH -e err_%j.txt
source activate tf
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 ./main.py