#!/bin/bash
# 启动超参网格训练（双卡，按 lr 分割）
# 用法: launch_grid.sh <gpu> <lr1> <lr2>
M=/mnt/mydisk/xinguan-brainai
GPU=$1
LR1=$2
LR2=$3

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=$M/hf
cd $M
exec ~/venvs/dl/bin/python $M/code/train_grid.py \
  --arch resnet18 --epochs 200 --seeds 0 1 2 \
  --lr-grid "$LR1" "$LR2" \
  --wd-grid 2e-4 1e-4 5e-5 2.5e-5 \
  --outdir $M/runs/ckpt_grid
