#!/usr/bin/env bash
# P4 pilot 第二批：3 卡并行（GPU0/2/3），每卡串行两个配置。
# 剂量定标依据 measure_grad_p4.py 实测（|g_CE|=20.4）：
#   whiten 0.249 -> {0.1(已完成), 1.0}
#   dis/con 6.0  -> {0.005, 0.05}（0.1 已证崩溃）
#   placebo 43.5 -> {0.01}（高档对照；若见信号再补 0.001）
# 卡内串行：
#   GPU0: disperse@0.005 -> whiten@1.0        （崩溃重跑低档 + whiten 剂量响应高档）
#   GPU2: concentrate@0.005 -> concentrate@0.05（阴性对照两档，剂量响应）
#   GPU3: placebo@0.01 -> disperse@0.05       （对照 + disperse 有效上限探测）
# 注意：不同 out 目录 = 独立 JSON，避免并发写竞争；各目录 --skip-base（base 读数已在 pilot JSON）。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python

chain () {  # gpu logtag "cmd1 && cmd2"
  local GPU=$1 LOGTAG=$2 CMDS=$3
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup bash -c "$CMDS" \
    > "logs/p4_b2_${LOGTAG}.log" 2>&1 &
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
$PY $B --out runs/v2_p4_forcing/pilot_b2_placebo_l0.01 --lambda 0.01 --kinds placebo && \
$PY $B --out runs/v2_p4_forcing/pilot_b2_disperse_l0.05 --lambda 0.05 --kinds disperse"
