#!/usr/bin/env bash
# P4 全量批 forcing（v2，base 批完成后发）。
# 矩阵：8 配方（E1 锚点清单）× 3 种子 × 4 条件 = 96 run。
# 分配：3 卡按种子分片，每卡 = 1 种子 × 4 条件串行进程 × 8 配方/进程 = 32 run，~14h。
# 断点续跑：每个 (seed,kind,lam) 一个 out JSON，按 (model,kind,lambda) 去重。
# base 对照：--skip-base（BASE_READ 仅锚点 2 配方）；其余配对由 merge 脚本
# join runs/v2_p4_forcing/full_base_lr*.json 完成。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python

MODELS8="resnet18_lr0.125_wd0.0001_s{S} resnet18_lr0.125_wd0.0002_s{S} resnet18_lr0.125_wd2.5e-05_s{S} resnet18_lr0.5_wd0.0001_s{S} resnet18_lr0.5_wd0.0002_s{S} resnet18_lr0.5_wd5e-05_s{S} resnet18_lr1.0_wd2.5e-05_s{S} resnet18_lr1.0_wd5e-05_s{S}"

chain () {
  local GPU=$1 SEED=$2
  local MODELS=${MODELS8//\{S\}/$SEED}
  local CMDS=""
  for cfg in "whiten 0.1" "whiten 1.0" "concentrate 0.005" "placebo 0.01"; do
    set -- $cfg
    CMDS="$CMDS
$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
  --out runs/v2_p4_forcing/full_s${SEED}_$1_lam$2 \
  --seed $SEED --kinds $1 --lambda $2 --models $MODELS"
  done
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup bash -c "$CMDS" \
    > "logs/p4_full_s${SEED}.log" 2>&1 &
  echo "gpu$GPU seed$SEED pid=$!"
}

chain 0 0
chain 2 1
chain 3 2
