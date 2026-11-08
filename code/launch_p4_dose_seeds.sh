#!/usr/bin/env bash
# P4 剂量扫描种子复制批（ADR-0016 附录二十五预注册，2026-11-08 19:44）。
# v3 修正：hi 链只跑 hi 模型（v2 的 --models 误含 lo，与 GPU0 lo 链重复白跑）。
# v2 修正：每 (种子,配方) 专属 out 目录 dose_seed_s{1,2}_{lo,hi}_lam{λ}，
# 保证单写者（v1 曾把 lo 模型误排进 hi 卡造成跨进程并发写同一 JSON）。
# 设计：2 配方 × 3λ{0.03,0.3,3.0} × 2 新种子{s1,s2} = 12 run。
#   λ=0.1/1.0 锚点 s1/s2 已在全量批（full_s{1,2}_whiten_lam{0.1,1.0}），不重跑。
# base：三种子 checkpoint 与读数均已在，--skip-base 安全跳过。
# 分配（load ~48，resnet18 实测 ~2GB）：
#   GPU0 = lo s1 3run → lo s2 3run（串行链）
#   GPU2 = hi s1 3run（与 llama-server 共存）
#   GPU3 = hi s2 3run（同上）
# 判定脚本按目录前缀 dose_seed_ 收数，s0 行仍在 dose_{lo,hi}_lam*。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python

launch () {
  local GPU=$1 SEED=$2 TAG=$3
  local CMDS=""
  for LAM in 0.03 0.3 3.0; do
    CMDS="$CMDS
$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
  --out runs/v2_p4_forcing/dose_seed_s${SEED}_${TAG}_lam${LAM} \
  --seed ${SEED} --kinds whiten --lambda ${LAM} \
  --models resnet18_lr0.125_wd0.0001_s${SEED}"
  done
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup bash -c "$CMDS" \
    > "logs/p4_dose_${TAG}_s${SEED}.log" 2>&1 &
  echo "gpu$GPU ${TAG} s${SEED} pid=$!"
}

# GPU0: lo s1 → lo s2 串行链
cat > /tmp/lo_chain.sh << INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
for SEED in 1 2; do
  for LAM in 0.03 0.3 3.0; do
    $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
      --out runs/v2_p4_forcing/dose_seed_s\${SEED}_lo_lam\${LAM} \
      --seed \${SEED} --kinds whiten --lambda \${LAM} \
      --models resnet18_lr1.0_wd5e-05_s\${SEED}
  done
done
INNER
chmod +x /tmp/lo_chain.sh
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/lo_chain.sh > logs/p4_dose_lo_s1s2.log 2>&1 &
echo "gpu0 lo s1+s2 pid=$!"

launch 2 1 hi
launch 3 2 hi
echo "all launched $(date)"
