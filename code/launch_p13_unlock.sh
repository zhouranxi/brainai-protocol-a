#!/usr/bin/env bash
# P13 跨剂量解锁批（ADR-0018 第七节冻结设计）：2 失灵源快照 × λ{0.3,1.0,3.0} × seed{201,202} = 12 run。
#   GPU0 = s0_seed7 源（6 run）  GPU2 = s0_seed13 源（6 run），墙钟 ~4h。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

make_chain() {  # $1=src-tag(seed7/seed13)  $2=snap  $3=marker
  cat > /tmp/p13_$1.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
SNAP=$2
for LAM in 0.3 1.0 3.0; do
  for FS in 201 202; do
    \$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \\
      --out runs/v2_p4_forcing/p13_unlock_$1_lam\${LAM}_seed\${FS} \\
      --seed \${FS} --kinds whiten --lambda \${LAM} \\
      --models resnet18_lr0.125_wd0.0001_s0 --resume-snap \${SNAP}
  done
done
echo done > $3
INNER
  chmod +x /tmp/p13_$1.sh
}

S7=runs/v2_p4_forcing/p12_m1a_s0_seed7_lam0.03/resnet18_lr0.125_wd0.0001_s0__whiten_lam0.03_ep40snap.pt
S13=runs/v2_p4_forcing/p12_m1a_s0_seed13_lam0.03/resnet18_lr0.125_wd0.0001_s0__whiten_lam0.03_ep40snap.pt

make_chain seed7 "$S7" /tmp/p13_done_g0
make_chain seed13 "$S13" /tmp/p13_done_g2

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p13_seed7.sh > logs/p13_g0.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p13_seed13.sh > logs/p13_g2.log 2>&1 < /dev/null &
B=$!
echo "P13: seed7src pid=$A (GPU0), seed13src pid=$B (GPU2)"
echo "launched $(date '+%F %T'); wall ~4h; markers: /tmp/p13_done_g{0,2}"
