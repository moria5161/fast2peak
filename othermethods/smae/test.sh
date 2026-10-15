#!/bin/bash
#SBATCH --job-name=smae
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=node5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

set -e

echo "--- 任务开始于: $(date) ---"


# python smae_test.py \
#     --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps1 mr0.1/epoch_21.pth" \
#     --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
#     --arch 'spectraMAE_base_ss1600_ps1' \
#     --mask_ratio 0.1 \
#     --batch_size 16 \
#     --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps1 mr0.1"

python smae_test.py \
    --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps1/epoch_300_final.pth" \
    --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
    --arch 'spectraMAE_base_ss1600_ps1' \
    --mask_ratio 0.5 \
    --batch_size 24 \
    --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps1"

# python smae_test.py \
#     --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps2/epoch_500.pth" \
#     --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
#     --arch 'spectraMAE_base_ss1600_ps2' \
#     --mask_ratio 0.5 \
#     --batch_size 64 \
#     --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps2"

# python smae_test.py \
#     --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps4/epoch_1000_final.pth" \
#     --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
#     --arch 'spectraMAE_base_ss1600_ps4' \
#     --mask_ratio 0.5 \
#     --batch_size 256 \
#     --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps4"


# python smae_test.py \
#     --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps8/epoch_1000_final.pth" \
#     --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
#     --arch 'spectraMAE_base_ss1600_ps8' \
#     --mask_ratio 0.5 \
#     --batch_size 1024 \
#     --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps8"

# python smae_test.py \
    # --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps16/epoch_1000_final.pth" \
    # --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
    # --arch 'spectraMAE_base_ss1600_ps16' \
    # --mask_ratio 0.5 \
    # --batch_size 1024 \
    # --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps16"

# python smae_test.py \
    # --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps32/epoch_1000_final.pth" \
    # --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
    # --arch 'spectraMAE_base_ss1600_ps32' \
    # --mask_ratio 0.5 \
    # --batch_size 1024 \
    # --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps32"

# python smae_test.py \
    # --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps64/epoch_1000_final.pth" \
    # --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
    # --arch 'spectraMAE_base_ss1600_ps64' \
    # --mask_ratio 0.5 \
    # --batch_size 1024 \
    # --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps64"

# python smae_test.py \
#     --chkpt_dir "/home/slurmusers/moria/codes/fast2peak_v2/smae/experiments/smae with train.h5 ss1600 ps100/epoch_1000_final.pth" \
#     --input_path "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
#     --arch 'spectraMAE_base_ss1600_ps100' \
#     --mask_ratio 0.5 \
#     --batch_size 1024 \
#     --output_dir "/home/slurmusers/moria/codes/fast2peak_v2/experiments/smae raw ps100"


echo "--- 任务结束于: $(date) ---"