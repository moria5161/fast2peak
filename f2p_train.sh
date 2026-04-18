#!/bin/bash
#SBATCH --job-name=f2p_train
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=node3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


cd /home/slurmusers/moria/codes/fast2peak_v2


# train

# torchrun --nproc_per_node=4  \
#         "main.py" \
#         --train \
#         --exp_name "training f2p rpe ps8 dp 12" \
#         --model_name "Fast2PeakRPE" \
#         --train_data_dir "datasets/train" \
#         --ddp \
#         --epochs 1000 \
#         --batch_size 384 \
#         --gamma 11 \
#         --mask_ratio 0.1 \
#         --patch_size 8 \
#         --depth 12 \
#         --workers 4



# fine tuning

torchrun --nproc_per_node=1  \
        "main.py" \
        --train \
        --exp_name "fine-tuning f2p rpe on graphene6 epoch100" \
        --model_name "Fast2PeakRPE" \
        --fine_tune_data_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/split/graphene6.h5" \
        --ddp \
        --lr 1e-6 \
        --epochs 100 \
        --batch_size 1 \
        --gamma 11 \
        --mask_ratio 0.05 \
        --patch_size 16 \
        --depth 8 \
        --workers 2 \
        --pretrained "/home/slurmusers/moria/codes/fast2peak_v2/experiments/training f2p rpe/checkpoints/best_checkpoint.pth"