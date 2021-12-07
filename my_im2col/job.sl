#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --gres=gpu:1
./im2col.out
