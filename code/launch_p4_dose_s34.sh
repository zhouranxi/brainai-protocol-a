#!/usr/bin/env bash
# P4 低剂量响应上限复制批（ADR-0016 附录二十六预注册，2026-11-09 18:40 跑前冻结）。
# 设计：先训 4 个 base（hi/lo × s3/s4，train_grid 200ep 协议）→ base 读数追加
#   → 12 个新 forcing run（2 配方 × s3/s4 × λ{0.03,0.3,3.0}）→ 自动 5 种子判定。
# 判据 R5-s/R6-s/R7-s/R8-s 见附录二十六，本脚本不做任何判定逻辑（只调脚本）。
# 资源：GPU0 串行单写者；每 (种子,配方) 专属 out 目录（沿用附录二十五约定）。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

CMDS="
# ---- stage 1: train 4 missing base checkpoints (train_grid skips existing) ----
$PY code/train_grid.py --arch resnet18 --epochs 200 --seeds 3 4 \
  --lr-grid 0.125 1.0 --wd-grid 0.0001 5e-05 \
  --outdir runs/ckpt_grid --device cuda

# ---- stage 2: append base readings for the 4 new models ----
$PY code/base_batch_p4.py --seed 0 \
  --out runs/v2_p4_forcing/full_base_lr0.125.json \
  --tags resnet18_lr0.125_wd0.0001_s3 resnet18_lr0.125_wd0.0001_s4
$PY code/base_batch_p4.py --seed 0 \
  --out runs/v2_p4_forcing/full_base_lr1.0.json \
  --tags resnet18_lr1.0_wd5e-05_s3 resnet18_lr1.0_wd5e-05_s4

# ---- stage 3: 12 forcing runs, serial, per-(seed,recipe) out dirs ----
"
for SEED in 3 4; do
  for TAG in lo hi; do
    if [ "$TAG" = lo ]; then MODEL="resnet18_lr1.0_wd5e-05_s${SEED}"; else MODEL="resnet18_lr0.125_wd0.0001_s${SEED}"; fi
    for LAM in 0.03 0.3 3.0; do
      CMDS="$CMDS
$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
  --out runs/v2_p4_forcing/dose_seed_s${SEED}_${TAG}_lam${LAM} \
  --seed ${SEED} --kinds whiten --lambda ${LAM} --models ${MODEL}"
    done
  done
done
CMDS="$CMDS

# ---- stage 4: fold s3/s4 into archive v2 + 5-seed verdict ----
$PY code/build_dose_scan_seeds.py --base runs/v2_p4_forcing \
  --out runs/v2_p4_forcing/dose_scan_seeds_v2.json
$PY code/dose_shape_fit_seeds.py --data runs/v2_p4_forcing/dose_scan_seeds_v2.json \
  2>&1 | tee runs/v2_p4_forcing/dose_s34_verdict.txt
"

cat > /tmp/p4_s34_chain.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
$CMDS
INNER
chmod +x /tmp/p4_s34_chain.sh
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p4_s34_chain.sh \
  > logs/p4_s34_chain.log 2>&1 < /dev/null &
echo "gpu0 s3/s4 chain pid=$! launched $(date)"
