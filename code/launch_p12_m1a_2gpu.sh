#!/usr/bin/env bash
# P12 M1a 收集批（ADR-0018 冻结设计）：s0/s3 各 8 run-seed (7..14)，λ=0.03，200ep。
# 仪表：run JSON 带 history（per-epoch forcing_loss）+ ep40 快照。
#   GPU0 = s0 链（8 run）  GPU2 = s3 链（8 run），墙钟 ~4.5h。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

make_chain() {  # $1=seed-tag  $2=marker
  cat > /tmp/p12_m1a_$1.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
for RS in 7 8 9 10 11 12 13 14; do
  \$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \\
    --out runs/v2_p4_forcing/p12_m1a_$1_seed\${RS}_lam0.03 \\
    --seed \${RS} --kinds whiten --lambda 0.03 \\
    --models resnet18_lr0.125_wd0.0001_$1
done
echo done > $2
INNER
  chmod +x /tmp/p12_m1a_$1.sh
}

make_chain s0 /tmp/p12_m1a_done_g0
make_chain s3 /tmp/p12_m1a_done_g2

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p12_m1a_s0.sh > logs/p12_m1a_g0.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p12_m1a_s3.sh > logs/p12_m1a_g2.log 2>&1 < /dev/null &
B=$!
echo "P12 M1a: s0 pid=$A (GPU0), s3 pid=$B (GPU2)"
echo "launched $(date '+%F %T'); wall ~4.5h; markers: /tmp/p12_m1a_done_g{0,2}"
