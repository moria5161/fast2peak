#!/bin/bash
#SBATCH --job-name=f2p_test
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=node4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


cd /home/slurmusers/moria/codes/fast2peak_v2

# test Simu
python "main.py" \
        --test \
        --test_datapath "/home/slurmusers/moria/codes/fast2peak_v2/datasets/test/Simu.h5" \
        --test_model_path "/home/slurmusers/moria/codes/fast2peak_v2/experiments/f2p blockmask 16/checkpoints/best_checkpoint.pth" \
        --test_batch_size 1024



# test SSEs
# python "main.py" \
#         --test \
#         --test_datapath "/home/slurmusers/moria/codes/fast2peak_v2/notebooks/figure2/Demo_specs.h5" \
#         --test_model_path "/home/slurmusers/moria/codes/fast2peak_v2/experiments/training resunet/checkpoints/best_checkpoint.pth" \
#         --test_batch_size 1024



# python "main.py" \
#         --test \
#         --test_datapath "/home/slurmusers/moria/codes/fast2peak_v2/datasets/anjisuan_cut" \
#         --test_model_path "/home/slurmusers/moria/codes/fast2peak_v2/experiments/training f2p rpe/checkpoints/best_checkpoint.pth" \
#         --test_batch_size 1024
