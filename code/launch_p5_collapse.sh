#!/usr/bin/env bash
# P5 塌陷概率量化批（附录三十三预注册，C1/C2 判读跑前冻结）。
# C1: s0 hi whiten@0.3 复跑 x run-seed {1..8}（原塌陷 run 是 seed 0）
# C2: 五种子 hi whiten@0.2 各一点（0.3 点已全库，不重跑）
# 资源：GPU0 串行 13 run ~7-9h。协议与附录二十三~三十逐字一致。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/p5_collapse_main.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

# ---- C1: s0 reruns, run-seed 1..8 ----
for RSEED in 1 2 3 4 5 6 7 8; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/collapse_c1_seed${RSEED} \
    --seed ${RSEED} --kinds whiten --lambda 0.3 \
    --models resnet18_lr0.125_wd0.0001_s0
done

# ---- C2: lambda=0.2 across five seeds ----
for S in 0 1 2 3 4; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/collapse_c2_s${S}_lam0.2 \
    --seed 0 --kinds whiten --lambda 0.2 \
    --models resnet18_lr0.125_wd0.0001_s${S}
done

echo "P5 collapse batch done" > /tmp/p5_collapse_done
INNER
chmod +x /tmp/p5_collapse_main.sh

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p5_collapse_main.sh \
  > logs/p5_collapse_g0.log 2>&1 < /dev/null &
echo "P5 chain pid=$!"
echo "launched $(date '+%F %T'); expected ~7-9h on GPU0 (13 runs serial)"
