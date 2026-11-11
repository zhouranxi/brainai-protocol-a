#!/usr/bin/env bash
# P7 E 臂：四种子带 run 级稳定性批（附录三十五冻结设计）。
# 10 run: s1/s2 各 run-seed 1 全剂量轴 lam {0.03,0.1,0.3,1.0,3.0}。
# GPU0 串行 ~5.5h。协议与附录二十三~三十四逐字一致。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/p7_e_main.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

for S in 1 2; do
  for LAM in 0.03 0.1 0.3 1.0 3.0; do
    $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
      --out runs/v2_p4_forcing/p7_e_s${S}_seed1_lam${LAM} \
      --seed 1 --kinds whiten --lambda ${LAM} \
      --models resnet18_lr0.125_wd0.0001_s${S}
  done
done

echo "P7 E arm done" > /tmp/p7_e_done
INNER
chmod +x /tmp/p7_e_main.sh

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p7_e_main.sh \
  > logs/p7_e_g0.log 2>&1 < /dev/null &
echo "P7 E chain pid=$!"
echo "launched $(date '+%F %T'); expected ~5.5h on GPU0 (10 runs serial)"
