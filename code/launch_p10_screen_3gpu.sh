#!/usr/bin/env bash
# P10 第一轮（筛查轮，ADR-0017 第六节冻结设计）：对称 5x4 池，44 run。
#   GPU0 = s3  3 条曲线（run-seed 3/4/5 × 5 λ）= 15 run
#   GPU2 = s4  3 条曲线（run-seed 3/4/5 × 5 λ）= 15 run
#   GPU3 = s0 run-seed3（4 run，λ0.3 复用 P5 c1_seed3）+ s1(5) + s2(5) = 14 run
# 产出 20 条全曲线判定池（5 种子 × 4 抽样）。墙钟 ~8.5h。
# 协议与 ADR-0016/0017 各批逐字一致。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

make_seedcur() {  # $1=seed-tag(s3/s4)  $2=GPU  $3=marker
  cat > /tmp/p10_chain_$1.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
for RS in 3 4 5; do
  for LAM in 0.03 0.1 0.3 1.0 3.0; do
    \$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \\
      --out runs/v2_p4_forcing/p10_$1_seed\${RS}_lam\${LAM} \\
      --seed \${RS} --kinds whiten --lambda \${LAM} \\
      --models resnet18_lr0.125_wd0.0001_$1
  done
done
echo done > $3
INNER
  chmod +x /tmp/p10_chain_$1.sh
}

cat > /tmp/p10_chain_mix.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
# s0 run-seed 3：4 run（lam 0.3 复用 P5 collapse_c1_seed3，不重跑）
for LAM in 0.03 0.1 1.0 3.0; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/p10_s0_seed3_lam${LAM} \
    --seed 3 --kinds whiten --lambda ${LAM} \
    --models resnet18_lr0.125_wd0.0001_s0
done
# s1 / s2 run-seed 3：各 5 run
for S in 1 2; do
  for LAM in 0.03 0.1 0.3 1.0 3.0; do
    $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
      --out runs/v2_p4_forcing/p10_s${S}_seed3_lam${LAM} \
      --seed 3 --kinds whiten --lambda ${LAM} \
      --models resnet18_lr0.125_wd0.0001_s${S}
  done
done
echo done > /tmp/p10_done_g3
INNER
chmod +x /tmp/p10_chain_mix.sh

make_seedcur s3 0 /tmp/p10_done_g0
make_seedcur s4 2 /tmp/p10_done_g2

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p10_chain_s3.sh > logs/p10_g0_s3.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p10_chain_s4.sh > logs/p10_g2_s4.log 2>&1 < /dev/null &
B=$!
CUDA_VISIBLE_DEVICES=3 setsid nohup bash /tmp/p10_chain_mix.sh > logs/p10_g3_mix.log 2>&1 < /dev/null &
C=$!
echo "P10 screen 3-GPU: s3 pid=$A (GPU0), s4 pid=$B (GPU2), mix pid=$C (GPU3)"
echo "launched $(date '+%F %T'); wall ~8.5h; markers: /tmp/p10_done_g{0,2,3}"
