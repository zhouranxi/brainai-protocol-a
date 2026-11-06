#!/usr/bin/env bash
# P4 pilot 两段启动：3ep 冒烟（全管线含端点测量，量单 epoch 时间）→ 通过后 200ep 正式。
# 用法：在服务器上 bash code/launch_p4_pilot.sh smoke | full
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
MODE=${1:-smoke}

if [ "$MODE" = "smoke" ]; then
  OUT=runs/v2_p4_forcing/smoke3
  EP=3
else
  OUT=runs/v2_p4_forcing/pilot
  EP=200
fi

mkdir -p "$OUT" logs
echo "[launch_p4] mode=$MODE out=$OUT epochs=$EP"
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY code/train_forcing_p4.py \
  --arm pilot --out "$OUT" --epochs "$EP" \
  > "logs/p4_${MODE}.log" 2>&1 &
echo "launched pid=$!"
sleep 2
tail -3 "logs/p4_${MODE}.log" || true
