#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --output=im2col.out
#SBATCH --error=im2col.err
#SBATCH --gres=gpu:1
./im2col
