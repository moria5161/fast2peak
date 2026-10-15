# SpectraMAENet
## About
This repository contains the code and resources of the following paper:
A spectral self-supervised learning method based on Mask AutoEncoders

## Overview of the framework
SMAE is a new self-supervised learning framework for Raman spectroscopy, which includes a transformer encoder structure and uses a masked pre-training strategy.

<p align="center">
<img  src="SMAE.jpg"> 
</p>

## **Setup environment**
Setup the required environment using `environment.yml` with Anaconda. While in the project directory run:

    conda env create
    
Activate the environment

    conda activate SMAE

## **Pre-train**
### Pretrain-bacteriaID

cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/smae/SpectraMAENet
python main_unsupervisePretrain.py --data_path ./datasets/bacteriaID/ --dataset reference --model 'spectraMAE_ss1000_ps100_dim256_h8' --device cuda:0 --epoch 500 --batch_size 64 --lr 1e-3 --wd 1e-5 --mask_ratio 0.5 --save_path './experiments/smae_raw/'


cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/smae/SpectraMAENet

python main_unsupervisePretrain.py --data_path /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/datasets/train/train.h5 --dataset reference --model 'spectraMAE_ss1600_ps100_dim256_h8' --device cuda:0 --epoch 1000 --batch_size 512 --lr 1e-3 --wd 1e-5 --mask_ratio 0.5 --save_path './experiments/smae with train.h5/'


cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/smae/SpectraMAENet

python main_unsupervisePretrain.py --data_path /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/datasets/train/train.h5 --dataset reference --model 'spectraMAE_base_ss1600_ps2' --device cuda:0 --epoch 1000 --batch_size 512 --lr 1e-3 --wd 1e-5 --mask_ratio 0.5 --save_path './experiments/smae with train.h5 ss1600 ps2/'


cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/smae/SpectraMAENet

python main_unsupervisePretrain.py --data_path /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/datasets/test/simu1024.h5 --dataset reference --model 'spectraMAE_base_ss1600_ps100' --device cuda:0 --epoch 1000 --batch_size 16 --lr 1e-3 --wd 1e-5 --mask_ratio 0.5 --save_path './experiments/smae on simu ps100/'

python main_unsupervisePretrain.py --data_path /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/datasets/test/simu1024.h5 --dataset reference --model 'spectraMAE_base_ss1600_ps16' --device cuda:0 --epoch 1000 --batch_size 16 --lr 1e-3 --wd 1e-5 --mask_ratio 0.5 --save_path './experiments/smae on simu ps16/'

python main_unsupervisePretrain.py --data_path /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/datasets/test/simu1024.h5 --dataset reference --model 'spectraMAE_base_ss1600_ps4' --device cuda:0 --epoch 1000 --batch_size 16 --lr 1e-3 --wd 1e-5 --mask_ratio 0.5 --save_path './experiments/smae on simu ps4/'

python main_unsupervisePretrain.py --data_path /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/5161-zhangnannan/fast2peak/datasets/test/simu1024.h5 --dataset reference --model 'spectraMAE_base_ss1600_ps2' --device cuda:0 --epoch 1000 --batch_size 16 --lr 1e-3 --wd 1e-5 --mask_ratio 0.5 --save_path './experiments/smae on simu ps2/'



### Pretrain-DEEPER
    python main_denoisingPretrain.py --data_path ./datasets/DEEPER/ --device cuda:0 --epoch 800 --batch_size 64 --lr 1e-3 --wd 1e-5 --mask_ratio 0.5 --save_path ./Pretrain_weight/

## **Finetune**
### Finetune-bacteriaID
    python main_finetune.py --data_path ./datasets/bacteriaID/ --dataset finetune --device cuda:0 --epoch 200 --batch_size 16 --lr 1e-4 --wd 1e-5 --save_path ./Finetune_weight/ --patience 20

## **Evaluation**
### Evaluation-bacteriaID
    jupyter 2_prediction.ipynb
### Evaluation-DEEPER
    python main_denoisingTest.py
## Citation
Ren P, Zhou R, Li Y. A Self-supervised Learning Method for Raman Spectroscopy based on Masked Autoencoders[J]. Expert Systems with Applications, 2025: 128576.

## Resources
Self-supervised learning uses spectral data as: Pathogen bacteria dataset [bacteria_ID](https://www.dropbox.com/scl/fo/fb29ihfnvishuxlnpgvhg/AJToUtts-vjYdwZGeqK4k-Y?rlkey=r4p070nsuei6qj3pjp13nwf6l&e=1&dl=0)

Denoising dataset: [DEEPER](https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EqZaY-_FrGdImybIGuMCvb8Bo_YD1Bc9ATBxbLxdDIv0RA?e=5%3aHhLp91&fromShare=true&at=9)
