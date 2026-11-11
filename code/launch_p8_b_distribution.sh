#!/usr/bin/env bash
# P8：b 的 run 级抽样分布批（ADR-0017 预注册，F1/F2/F3 判读跑前冻结）。
# 14 run：s0 λ{0.03,0.1,1.0,3.0}(4) + s1 全 5 点(5) + s2 全 5 点(5)，全 run-seed 2。
# s0 的 λ=0.3 第三点取 P5 collapse_c1_seed2（ADR-0017 选样注记，跑前冻结）。
# GPU0 串行 ~7.7h。协议与 ADR-0016 附录二十三~三十五逐字一致。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/p8_main.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

# s0: 4 run (lam 0.3 取自 P5 seed2, 不重跑)
for LAM in 0.03 0.1 1.0 3.0; do
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/p8_s0_seed2_lam${LAM} \
    --seed 2 --kinds whiten --lambda ${LAM} \
    --models resnet18_lr0.125_wd0.0001_s0
done

# s1 / s2: 各 5 run 全剂量轴
for S in 1 2; do
  for LAM in 0.03 0.1 0.3 1.0 3.0; do
    $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
      --out runs/v2_p4_forcing/p8_s${S}_seed2_lam${LAM} \
      --seed 2 --kinds whiten --lambda ${LAM} \
      --models resnet18_lr0.125_wd0.0001_s${S}
  done
done

echo "P8 done" > /tmp/p8_done
INNER
chmod +x /tmp/p8_main.sh

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p8_main.sh \
  > logs/p8_g0.log 2>&1 < /dev/null &
echo "P8 chain pid=$!"
echo "launched $(date '+%F %T'); expected ~7.7h on GPU0 (14 runs serial)"
