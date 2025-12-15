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


DATA_CFG=./configs/datasets/domainnet.yaml

# Base Line
python main.py --data_cfg $DATA_CFG \
               --train_cfg ./configs/trainers/bimc.yaml

python main.py --data_cfg $DATA_CFG \
               --train_cfg ./configs/trainers/bimc_ensemble.yaml

# Edge
python main.py --data_cfg $DATA_CFG \
               --train_cfg ./configs/trainers/edge_meta.yaml \
               --hyperparam_sweep

# Prompt
python main.py --data_cfg $DATA_CFG \
               --train_cfg ./configs/trainers/bimc_prompt.yaml \
               --prompt \
#               --prompt_checkpoint outputs/prompts_latest.pth

# Edge + Prompt
python main.py --data_cfg $DATA_CFG \
               --train_cfg ./configs/trainers/edge_meta.yaml \
               --hyperparam_sweep \
               --prompt \
#               --prompt_checkpoint outputs/prompts_latest.pth