#!/bin/bash
#SBATCH --job-name=bimc3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=12G
#SBATCH --time=2-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err


# Ours

#python main.py --data_cfg ./configs/datasets/domainnet.yaml --train_cfg ./configs/trainers/bimc.yaml

#python main.py --data_cfg ./configs/datasets/domainnet.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

#python main.py --data_cfg ./configs/datasets/domainnet.yaml --train_cfg ./configs/trainers/edge.yaml --hyperparam_sweep
#
#python main.py --data_cfg ./configs/datasets/domainnet.yaml --train_cfg ./configs/trainers/edge_ensemble.yaml --hyperparam_sweep

python main.py --data_cfg ./configs/datasets/domainnet.yaml --train_cfg ./configs/trainers/edge_meta3.yaml --meta

#python main.py --data_cfg ./configs/datasets/domainnet.yaml --train_cfg ./configs/trainers/edge_meta.yaml --meta --router_checkpoint outputs/router.pth