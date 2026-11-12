#!/usr/bin/env bash
# P10 第二轮（解剖轮，ADR-0017 第七节冻结）：钉死 s0 λ=0.03 失灵的可复现性。
# 5 run：s0@0.03 × run-seed {3,4,5,6}（seed3 为确定性复刻对照）+ s2@0.03 × seed3。
# GPU0 串行 ~2.8h。H1/H2 判读跑前冻结（ADR-0017 第七节）。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/p10_r2_main.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

for RS in 3 4 5 6; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/p10r2_s0_seed${RS}_lam0.03 \
    --seed ${RS} --kinds whiten --lambda 0.03 \
    --models resnet18_lr0.125_wd0.0001_s0
done

$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
  --out runs/v2_p4_forcing/p10r2_s2_seed3_lam0.03 \
  --seed 3 --kinds whiten --lambda 0.03 \
  --models resnet18_lr0.125_wd0.0001_s2

echo "P10 round 2 done" > /tmp/p10r2_done
INNER
chmod +x /tmp/p10_r2_main.sh

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p10_r2_main.sh \
  > logs/p10r2_g0.log 2>&1 < /dev/null &
echo "P10-R2 chain pid=$!"
echo "launched $(date '+%F %T'); expected ~2.8h on GPU0 (5 runs serial)"
