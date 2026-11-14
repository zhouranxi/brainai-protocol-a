#!/usr/bin/env bash
# P12 M1b 分叉批（ADR-0018 冻结条款触发）：失灵源 s0_seed7 + 健康源 s0_seed8 各分叉 4 个（--seed 101..104）。
# 从 ep40 快照 --resume-snap 续训，协议同源（200ep 计，快照已含 40ep）。GPU0/GPU2 双链，~4.5h。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/p12_m1b_a.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
SNAP=runs/v2_p4_forcing/p12_m1a_s0_seed7_lam0.03/resnet18_lr0.125_wd0.0001_s0__whiten_lam0.03_ep40snap.pt
for FS in 101 102 103 104; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/p12_m1b_failsrc_seed${FS} \
    --seed ${FS} --kinds whiten --lambda 0.03 \
    --models resnet18_lr0.125_wd0.0001_s0 --resume-snap ${SNAP}
done
echo done > /tmp/p12_m1b_done_g0
INNER

cat > /tmp/p12_m1b_b.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
SNAP=runs/v2_p4_forcing/p12_m1a_s0_seed8_lam0.03/resnet18_lr0.125_wd0.0001_s0__whiten_lam0.03_ep40snap.pt
for FS in 101 102 103 104; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/p12_m1b_healthysrc_seed${FS} \
    --seed ${FS} --kinds whiten --lambda 0.03 \
    --models resnet18_lr0.125_wd0.0001_s0 --resume-snap ${SNAP}
done
echo done > /tmp/p12_m1b_done_g2
INNER

chmod +x /tmp/p12_m1b_a.sh /tmp/p12_m1b_b.sh
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p12_m1b_a.sh > logs/p12_m1b_g0.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p12_m1b_b.sh > logs/p12_m1b_g2.log 2>&1 < /dev/null &
B=$!
echo "P12 M1b: failsrc pid=$A (GPU0), healthysrc pid=$B (GPU2)"
echo "launched $(date '+%F %T'); wall ~4.5h; markers: /tmp/p12_m1b_done_g{0,2}"
