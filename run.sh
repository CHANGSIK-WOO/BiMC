#!/bin/bash
#SBATCH --job-name=bimc
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

python main.py --data_cfg ./configs/datasets/domainnet.yaml --train_cfg ./configs/trainers/bimc.yaml

## CIFAR BIMC
#python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc.yaml
#
## CIFAR BIMC_Ensemble
#python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml
#
## MiniImagenet BIMC
#python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc.yaml
#
## MiniImagenet BIMC_Ensemble
#python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml
#
## CUB200 BIMC
#python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml
#
## CUB200 BIMC_Ensemble
#python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml