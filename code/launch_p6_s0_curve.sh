#!/usr/bin/env bash
# P6 D 臂：s0 复曲线 run-seed 1 全剂量轴（附录三十四冻结设计）。
# 4 run: lam {0.03, 0.1, 1.0, 3.0}，与已有 collapse_c1_seed1 (lam 0.3)
# 拼成 s0 第二条完整 5 点曲线。GPU0 串行 ~2.2-3.3h。
# 协议与附录二十三~三十三逐字一致。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/p6_d_main.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

for LAM in 0.03 0.1 1.0 3.0; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/p6_d_s0_seed1_lam${LAM} \
    --seed 1 --kinds whiten --lambda ${LAM} \
    --models resnet18_lr0.125_wd0.0001_s0
done

echo "P6 D arm done" > /tmp/p6_d_done
INNER
chmod +x /tmp/p6_d_main.sh

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p6_d_main.sh \
  > logs/p6_d_g0.log 2>&1 < /dev/null &
echo "P6 D chain pid=$!"
echo "launched $(date '+%F %T'); expected ~2.2-3.3h on GPU0 (4 runs serial)"
