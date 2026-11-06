#!/usr/bin/env bash
# P4 pilot 第二批 v2（幂迭代 autograd NaN 毒化修复后重发）。
# 修复内容：特征向量求解 no_grad+float64+CPU eigh，梯度走 Rayleigh 商（有界）；
#          训练环 NaN/Inf 护栏（坏 batch 退化纯 CE 并计数 n_bad_fl）。
# λ 阶梯不变（重测 |g|：whiten 0.254 / dis-con 5.21 / placebo 51.2，CE 19.3）。
# 布局（3 卡并行，每卡串行两个配置，~2.5h）：
#   GPU0: disperse@0.005 -> whiten@1.0
#   GPU2: concentrate@0.005 -> concentrate@0.05
#   GPU3: disperse@0.1（毒化归因复核：修好后若 0.1 不再崩 ⇒ 崩溃=NaN 缺陷而非剂量）
#         （placebo@0.01 在 GPU3 旧进程仍在跑，健康，不重发）
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python

chain () {
  local GPU=$1 LOGTAG=$2 CMDS=$3
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup bash -c "$CMDS" \
    > "logs/p4_b2v2_${LOGTAG}.log" 2>&1 &
  echo "gpu$GPU [$LOGTAG] pid=$!"
}

B="code/train_forcing_p4.py --arm pilot --epochs 200 --skip-base"

chain 0 gpu0 "\
$PY $B --out runs/v2_p4_forcing/pilot_b2_disperse_l0.005 --lambda 0.005 --kinds disperse && \
$PY $B --out runs/v2_p4_forcing/pilot_b2_whiten_l1.0    --lambda 1.0   --kinds whiten"

chain 2 gpu2 "\
$PY $B --out runs/v2_p4_forcing/pilot_b2_concentrate_l0.005 --lambda 0.005 --kinds concentrate && \
$PY $B --out runs/v2_p4_forcing/pilot_b2_concentrate_l0.05  --lambda 0.05  --kinds concentrate"

chain 3 gpu3 "\
$PY $B --out runs/v2_p4_forcing/pilot_b2_disperse_l0.1 --lambda 0.1 --kinds disperse"
