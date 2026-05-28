#!/bin/bash
#SBATCH --job-name=f2p_train
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=node5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=32G


cd /home/slurmusers/moria/codes/fast2peak_v2


# train

torchrun --nproc_per_node=4  \
        "main.py" \
        --train \
        --exp_name "f2p blockmask 16" \
        --model_name "Fast2PeakRPE" \
        --train_data_dir "datasets/train" \
        --ddp \
        --epochs 1000 \
        --batch_size 512 \
        --gamma 11 \
        --mask_ratio 0.1 \
        --block_size 16 \
        --patch_size 16 \
        --depth 8 \
        --workers 4


# torchrun --nproc_per_node=4  \
#         "main.py" \
#         --train \
#         --exp_name "ramanmae mr0.1 ps2" \
#         --model_name "RamanMAE" \
#         --train_data_dir "datasets/train" \
#         --ddp \
#         --epochs 1000 \
#         --batch_size 16 \
#         --gamma 11 \
#         --mask_ratio 0.1 \
#         --patch_size 1 \
#         --depth 8 \
#         --workers 4


# torchrun --nproc_per_node=2  \
#         "main.py" \
#         --train \
#         --exp_name "f2p rpe ps2 dp4 with duotai" \
#         --model_name "Fast2PeakRPE" \
#         --train_data_dir "datasets/duotai.h5" \
#         --ddp \
#         --epochs 1000 \
#         --batch_size 4 \
#         --gamma 3 \
#         --mask_ratio 0.5 \
#         --patch_size 2 \
#         --depth 4 \
#         --spectrum_length 2048 \
#         --workers 1


# fine tuning

# torchrun --nproc_per_node=2  \
#         "main.py" \
#         --train \
#         --exp_name "fine-tuning f2p rpe on ps8 duotai epoch100" \
#         --model_name "Fast2PeakRPE" \
#         --fine_tune_data_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/duotai.h5" \
#         --ddp \
#         --lr 1e-6 \
#         --epochs 100 \
#         --batch_size 1 \
#         --gamma 3 \
#         --mask_ratio 0.05 \
#         --patch_size 8 \
#         --depth 12 \
#         --workers 2 \
#         --pretrained "/home/slurmusers/moria/codes/fast2peak_v2/experiments/training f2p rpe ps8 dp12/checkpoints/best_checkpoint.pth"

# torchrun --nproc_per_node=2  \
#         "main.py" \
#         --train \
#         --exp_name "fine-tuning f2p rpe on ps8 anjisuan epoch100" \
#         --model_name "Fast2PeakRPE" \
#         --fine_tune_data_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/anjisuan.h5" \
#         --ddp \
#         --lr 1e-6 \
#         --epochs 100 \
#         --batch_size 1 \
#         --gamma 11 \
#         --mask_ratio 0.1 \
#         --patch_size 8 \
#         --depth 12 \
#         --workers 2 \
#         --pretrained "/home/slurmusers/moria/codes/fast2peak_v2/experiments/training f2p rpe/checkpoints/best_checkpoint.pth"