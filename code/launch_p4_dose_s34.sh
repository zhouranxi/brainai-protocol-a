#!/usr/bin/env bash
# P4 s3/s4 低剂量复制批 v2（附录二十六预注册，判据跑前冻结不变；用户 18:45 批准 GPU 0/2/3）。
# 与 v1 的差别：base 训练三卡并行，forcing 12 run 三链并行，末尾自动 v2 归档+判定。
# 分配（llama-server 与 resnet18 共存，resnet18 仅 ~2GB）：
#   GPU0 = 训 hi s3+s4 base → 读数 barrier → hi s3 3run → (排队) hi s4 3run
#   GPU2 = 训 lo s3 base    → 读数 barrier → lo s3 3run → lo s4 3run
#   GPU3 = 训 lo s4 base    → (空转待命；链都单写者不并发写)
# 判据 R5-s/R6-s/R7-s/R8-s 冻结于附录二十六，本脚本不做判定逻辑（只调脚本）。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

# ---------- stage A: base training, three GPUs in parallel ----------
cat > /tmp/p4s34_base_g0.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
$PY code/train_grid.py --arch resnet18 --epochs 200 --seeds 3 4 \
  --lr-grid 0.125 --wd-grid 0.0001 --outdir runs/ckpt_grid --device cuda
INNER
cat > /tmp/p4s34_base_g2.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
$PY code/train_grid.py --arch resnet18 --epochs 200 --seeds 3 \
  --lr-grid 1.0 --wd-grid 5e-05 --outdir runs/ckpt_grid --device cuda
INNER
cat > /tmp/p4s34_base_g3.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
$PY code/train_grid.py --arch resnet18 --epochs 200 --seeds 4 \
  --lr-grid 1.0 --wd-grid 5e-05 --outdir runs/ckpt_grid --device cuda
INNER
chmod +x /tmp/p4s34_base_g*.sh
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p4s34_base_g0.sh > logs/p4s34_base_g0.log 2>&1 < /dev/null &
P0=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p4s34_base_g2.sh > logs/p4s34_base_g2.log 2>&1 < /dev/null &
P2=$!
CUDA_VISIBLE_DEVICES=3 setsid nohup bash /tmp/p4s34_base_g3.sh > logs/p4s34_base_g3.log 2>&1 < /dev/null &
P3=$!
echo "base training pids: g0=$P0 g2=$P2 g3=$P3"

# ---------- stage B: barrier -> base readings -> marking file ----------
cat > /tmp/p4s34_main.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

for f in resnet18_lr0.125_wd0.0001_s3 resnet18_lr0.125_wd0.0001_s4 \
         resnet18_lr1.0_wd5e-05_s3 resnet18_lr1.0_wd5e-05_s4; do
  while [ ! -f "runs/ckpt_grid/${f}.pt" ]; do sleep 60; done
  echo "[barrier] ckpt ready: $f"
done
# 等最后一个 ckpt 写完（train_grid 每 20ep 覆写一次，末 epoch 后立即落 final）
sleep 120

$PY code/base_batch_p4.py --seed 0 \
  --out runs/v2_p4_forcing/full_base_lr0.125.json \
  --tags resnet18_lr0.125_wd0.0001_s3 resnet18_lr0.125_wd0.0001_s4
$PY code/base_batch_p4.py --seed 0 \
  --out runs/v2_p4_forcing/full_base_lr1.0.json \
  --tags resnet18_lr1.0_wd5e-05_s3 resnet18_lr1.0_wd5e-05_s4
echo "[barrier] base readings done"
touch /tmp/p4s34_bases_ready
INNER
chmod +x /tmp/p4s34_main.sh
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p4s34_main.sh > logs/p4s34_base_readings.log 2>&1 < /dev/null &
echo "barrier+readings pid=$!"

# ---------- stage C: three forcing chains, each single-writer ----------
mk_chain () {  # $1=GPU $2=CHAIN_ID $3=seed $4=recipe $5=model $6=done_marker $7=wait_marker
  local WAIT_CMD=""
  if [ -n "$7" ]; then WAIT_CMD="while [ ! -f $7 ]; do sleep 120; done"; fi
  cat > /tmp/p4s34_chain_$2.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
$WAIT_CMD
while [ ! -f /tmp/p4s34_bases_ready ]; do sleep 60; done
for LAM in 0.03 0.3 3.0; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/dose_seed_s$3_$4_lam\${LAM} \
    --seed $3 --kinds whiten --lambda \${LAM} --models $5
done
echo "chain $2 done" > $6
INNER
  chmod +x /tmp/p4s34_chain_$2.sh
  CUDA_VISIBLE_DEVICES=$1 setsid nohup bash /tmp/p4s34_chain_$2.sh \
    > logs/p4s34_chain_$2.log 2>&1 < /dev/null &
  echo "gpu$1 chain $2 pid=$!"
}
mk_chain 0 hi_s3 3 hi resnet18_lr0.125_wd0.0001_s3 /tmp/p4s34_done_hi_s3 ""
mk_chain 2 lo_s3s4 3 lo resnet18_lr1.0_wd5e-05_s3 /tmp/p4s34_done_lo_s3 ""
# GPU2 的 lo_s3s4 链接着跑 lo s4（同卡同配方族，串行单写者）：
cat > /tmp/p4s34_chain_lo_s4.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
while [ ! -f /tmp/p4s34_done_lo_s3 ]; do sleep 120; done
for LAM in 0.03 0.3 3.0; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/dose_seed_s4_lo_lam\${LAM} \
    --seed 4 --kinds whiten --lambda \${LAM} \
    --models resnet18_lr1.0_wd5e-05_s4
done
echo "chain lo_s4 done" > /tmp/p4s34_done_lo_s4
INNER
chmod +x /tmp/p4s34_chain_lo_s4.sh
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p4s34_chain_lo_s4.sh \
  > logs/p4s34_chain_lo_s4.log 2>&1 < /dev/null &
echo "gpu2 chain lo_s4 (deferred) pid=$!"
# GPU3 跑 lo_s4? 不：GPU3 的 base 训完即空。把 hi_s4 放 GPU3（等 hi_s3 无必要，
# 因为不在同一张卡；只需等 bases_ready）：
mk_chain 3 hi_s4 4 hi resnet18_lr0.125_wd0.0001_s4 /tmp/p4s34_done_hi_s4 ""

# ---------- stage D: finalize (archive v2 + 5-seed verdict) ----------
cat > /tmp/p4s34_finalize.sh <<INNER
#!/usr/bin/env bash
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
while ! ( [ -f /tmp/p4s34_done_hi_s3 ] && [ -f /tmp/p4s34_done_lo_s3 ] && \
          [ -f /tmp/p4s34_done_lo_s4 ] && [ -f /tmp/p4s34_done_hi_s4 ] ); do
  sleep 300
done
$PY code/build_dose_scan_seeds.py --base runs/v2_p4_forcing \
  --out runs/v2_p4_forcing/dose_scan_seeds_v2.json
$PY code/dose_shape_fit_seeds.py \
  --data runs/v2_p4_forcing/dose_scan_seeds_v2.json \
  2>&1 | tee runs/v2_p4_forcing/dose_s34_verdict.txt
INNER
chmod +x /tmp/p4s34_finalize.sh
setsid nohup bash /tmp/p4s34_finalize.sh > logs/p4s34_finalize.log 2>&1 < /dev/null &
echo "finalize watcher pid=$!"
echo "all launched $(date)"
