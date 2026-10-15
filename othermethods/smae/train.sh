#!/bin/bash
#SBATCH --job-name=smae
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=node7
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=36G

set -e

echo "--- 任务开始于: $(date) ---"

# python main_unsupervisePretrain.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/duotai.h5 --model 'spectraMAE_ss2048_ps16_dim256_h8' --device cuda:0 --epoch 1000 --batch_size 8 --lr 1e-3 --wd 1e-5 --mask_ratio 0.25 --save_path './experiments/smae with duotai ss2048 ps16/'


# torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps1_dim256_h8' --device cuda:0 --epoch 500 --batch_size 8 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps1 mr0.1/'

# torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps2_dim256_h8' --device cuda:0 --epoch 500 --batch_size 16 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps2 mr0.1/'

# torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps2_dim256_h8' --device cuda:0 --epoch 500 --batch_size 16 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps2 mr0.1/' --resume './experiments/smae with train.h5 ss1600 ps2 mr0.1/epoch_7.pth'

torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps4_dim256_h8' --device cuda:0 --epoch 500 --batch_size 64 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps4 mr0.1/' --resume './experiments/smae with train.h5 ss1600 ps4 mr0.1/epoch_48.pth'

# torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps4_dim256_h8' --device cuda:0 --epoch 500 --batch_size 64 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps4 mr0.1/'

# torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps8_dim256_h8' --device cuda:0 --epoch 500 --batch_size 512 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps8 mr0.1/'

# torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps16_dim256_h8' --device cuda:0 --epoch 500 --batch_size 256 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps16 mr0.1/'

# torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps32_dim256_h8' --device cuda:0 --epoch 500 --batch_size 512 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps32 mr0.1/'

# torchrun --nproc_per_node=4 main_unsupervisePretrain_ddp.py --data_path /home/slurmusers/moria/codes/fast2peak_v2/datasets/train/train.h5 --model 'spectraMAE_ss1600_ps64_dim256_h8' --device cuda:0 --epoch 500 --batch_size 256 --lr 1e-4 --wd 1e-5 --mask_ratio 0.1 --save_path './experiments/smae with train.h5 ss1600 ps64 mr0.1/'


echo "--- 任务结束于: $(date) ---"