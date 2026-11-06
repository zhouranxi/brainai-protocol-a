#!/usr/bin/env bash
# P4 全量批 base 读数 v2：24 模型（8 配方 × 3 种子）按 lr 族三卡分片，probe seed=0 统一。
# 与 E1 锚点可比（E1 base 读数即 seed0 探针）；断点续跑按 model 去重。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python

chain () {
  local GPU=$1 SHARD=$2 OUT=$3; shift 3
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $PY code/base_batch_p4.py \
    --tags "$@" --seed 0 --out "$OUT" \
    > "logs/p4_full_base_${SHARD}.log" 2>&1 &
  echo "gpu$GPU [$SHARD] pid=$! -> $OUT"
}

chain 0 lr0.125 runs/v2_p4_forcing/full_base_lr0.125.json \
  resnet18_lr0.125_wd0.0001_s0 resnet18_lr0.125_wd0.0001_s1 resnet18_lr0.125_wd0.0001_s2 \
  resnet18_lr0.125_wd0.0002_s0 resnet18_lr0.125_wd0.0002_s1 resnet18_lr0.125_wd0.0002_s2 \
  resnet18_lr0.125_wd2.5e-05_s0 resnet18_lr0.125_wd2.5e-05_s1 resnet18_lr0.125_wd2.5e-05_s2 \
  resnet18_lr0.125_wd5e-05_s0 resnet18_lr0.125_wd5e-05_s1 resnet18_lr0.125_wd5e-05_s2

chain 2 lr0.5 runs/v2_p4_forcing/full_base_lr0.5.json \
  resnet18_lr0.5_wd0.0001_s0 resnet18_lr0.5_wd0.0001_s1 resnet18_lr0.5_wd0.0001_s2 \
  resnet18_lr0.5_wd0.0002_s0 resnet18_lr0.5_wd0.0002_s1 resnet18_lr0.5_wd0.0002_s2 \
  resnet18_lr0.5_wd2.5e-05_s0 resnet18_lr0.5_wd2.5e-05_s1 resnet18_lr0.5_wd2.5e-05_s2 \
  resnet18_lr0.5_wd5e-05_s0 resnet18_lr0.5_wd5e-05_s1 resnet18_lr0.5_wd5e-05_s2

chain 3 lr1.0 runs/v2_p4_forcing/full_base_lr1.0.json \
  resnet18_lr1.0_wd0.0001_s0 resnet18_lr1.0_wd0.0001_s1 resnet18_lr1.0_wd0.0001_s2 \
  resnet18_lr1.0_wd0.0002_s0 resnet18_lr1.0_wd0.0002_s1 resnet18_lr1.0_wd0.0002_s2 \
  resnet18_lr1.0_wd2.5e-05_s0 resnet18_lr1.0_wd2.5e-05_s1 resnet18_lr1.0_wd2.5e-05_s2 \
  resnet18_lr1.0_wd5e-05_s0 resnet18_lr1.0_wd5e-05_s1 resnet18_lr1.0_wd5e-05_s2
