#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --output=winograd.out
#SBATCH --error=winograd.err
#SBATCH --gres=gpu:1
./Test