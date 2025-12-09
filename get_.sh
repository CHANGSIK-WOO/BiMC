#!/bin/bash

NUM_GPU=${1:-1}
MEM_PER_GPU=12
MEM=$(($NUM_GPU * $MEM_PER_GPU))

srun \
  -w vgi1 \
  -p debug \
  --gres=gpu:$NUM_GPU \
  --cpus-per-gpu=12 \
  --mem=${MEM}G \
  --pty $SHELL
