#!/usr/bin/env bash
# P4 s3/s4 锚点补齐批（附录三十预注册，A1/A2/A3 判读跑前冻结）。
# 与附录二十六批的差别：base checkpoint 与读数已在附录二十八批落盘，
# 无 base 训练、无读数 barrier——直接 4 个 forcing run（GPU0 串行）。
# 资源：GPU0 全空（llama-server 在 GPU2/3，Stu1 在 GPU1，均不碰）。
# 判读 A1/A2/A3 冻结于附录三十；本脚本不做判定逻辑，跑完 marker。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/p4_anchor_main.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

run () {  # $1=seed $2=recipe $3=model $4=lam
  $PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \
    --out runs/v2_p4_forcing/dose_seed_s$1_$2_lam$4 \
    --seed $1 --kinds whiten --lambda $4 --models $3
}

# hi s3 -> hi s4 -> lo s3 -> lo s4，每配方内 lambda 0.1 -> 1.0
for LAM in 0.1 1.0; do
  run 3 hi resnet18_lr0.125_wd0.0001_s3 $LAM
done
for LAM in 0.1 1.0; do
  run 4 hi resnet18_lr0.125_wd0.0001_s4 $LAM
done
for LAM in 0.1 1.0; do
  run 3 lo resnet18_lr1.0_wd5e-05_s3 $LAM
done
for LAM in 0.1 1.0; do
  run 4 lo resnet18_lr1.0_wd5e-05_s4 $LAM
done

echo "anchor batch done" > /tmp/p4_anchor_done
INNER
chmod +x /tmp/p4_anchor_main.sh

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p4_anchor_main.sh \
  > logs/p4_anchor_g0.log 2>&1 < /dev/null &
echo "anchor chain pid=$!"
echo "launched $(date '+%F %T'); expected ~2.5-3.5h on GPU0 (4 runs serial)"
