#!/bin/bash
#SBATCH --job-name=p2p
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


# Submit with sbatch from the repository root.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}" || exit 1


# python othermethods/peak2peak/p2p.py \
#     --data_dir "datasets/boundary" \
#     --data_savedir "experiments/eval p2p wd2" \
#     --epochs 20 \
#     --lr 1e-4 \
#     --batch_size 1 \
#     --weight_decay 2 \
#     --seed 615 \
#     --max_workers 10 \
#     --length 1024 \
#     --extend-by 32 \
#     --data_aug "add_raman_peaks" \
#     --model_name "FCN" \
#     --kernel_size 7


# Original P2P.
python othermethods/peak2peak/p2p_raw.py --input_dir "datasets/test" --output_dir "experiments/p2p_raw"
