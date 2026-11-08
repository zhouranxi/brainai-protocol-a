#!/usr/bin/env bash
# P4 剂量扫描（ADR-0016 附录二十三预注册，2026-11-08）。
# 目的：凹饱和响应面形态刻画（对数/幂律/线性）。
# 设计：2 配方（pilot 双锚点）× 3 新 λ {0.03, 0.3, 3.0} = 6 新 run。
#   λ=0.1/1.0 两点已在全量批（full_s0_whiten_lam{0.1,1.0}），不重跑。
# 分配：GPU0 = lr1.0_wd5e-05_s0 系 3 run；GPU2 = lr0.125_wd0.0001_s0 系 3 run。
#   GPU3 空置备用。
# 协议：与全量批逐字一致（200ep、seed 0、--skip-base、同测量链）。
# 断点续跑：每个 (model,kind,lambda) 一个 out JSON，键去重。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python

M_LO="resnet18_lr1.0_wd5e-05_s0"       # 低 D 基线锚点
M_HI="resnet18_lr0.125_wd0.0001_s0"    # 高 D 基线锚点

chain () {
  local GPU=$1 MODEL=$2 TAG=$3
  local CMDS=""
  for LAM in 0.03 0.3 3.0; do
    CMDS="$CMDS
$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
  --out runs/v2_p4_forcing/dose_${TAG}_lam${LAM} \
  --seed 0 --kinds whiten --lambda ${LAM} --models ${MODEL}"
  done
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup bash -c "$CMDS" \
    > "logs/p4_dose_${TAG}.log" 2>&1 &
  echo "gpu$GPU ${TAG} pid=$!"
}

chain 0 $M_LO lo          # lr1.0_wd5e-05_s0
chain 2 $M_HI hi          # lr0.125_wd0.0001_s0
