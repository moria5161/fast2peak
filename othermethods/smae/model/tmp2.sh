#!/bin/bash
#SBATCH --job-name=f2p_train
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=node5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -e

echo "--- 任务开始于: $(date) ---"


python "tmp2.py"