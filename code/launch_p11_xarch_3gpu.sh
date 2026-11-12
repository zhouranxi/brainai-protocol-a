#!/usr/bin/env bash
# P11 跨架构重尾复现批（ADR-0017 第八节冻结设计）：4 架构 × 2 曲线 × 5λ = 48 run。
#   GPU0 = resnet34 + regnet（16 run）
#   GPU2 = resnet50（12 run）+ base 重测
#   GPU3 = shufflenet（12 run）
# base 读数：--skip-base 模式下 forcing 臂 run JSON 自带 base 段（load_base_readings 现场重测）。
# 墙钟 ~4.5-6h。协议与 P5-P10 逐字一致。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

make_chain() {  # $1=tag  $2=GPU  $3=models空格串  $4=marker
  cat > /tmp/p11_chain_$1.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
for M in $3; do
  for RS in 1 2; do
    for LAM in 0.03 0.1 0.3 1.0 3.0; do
      \$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \\
        --out runs/v2_p4_forcing/p11_\${M}_seed\${RS}_lam\${LAM} \\
        --seed \${RS} --kinds whiten --lambda \${LAM} \\
        --models \${M}
    done
  done
done
echo done > $4
INNER
  chmod +x /tmp/p11_chain_$1.sh
}

make_chain a "0" "resnet34_lr0.125_wd5e-05_s0 regnet_y_400mf_lr0.125_wd5e-05_s0" /tmp/p11_done_g0
make_chain b "2" "resnet50_lr0.125_wd5e-05_s0" /tmp/p11_done_g2
make_chain c "3" "shufflenet_v2_x1_0_lr0.125_wd5e-05_s0" /tmp/p11_done_g3

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p11_chain_a.sh > logs/p11_g0.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p11_chain_b.sh > logs/p11_g2.log 2>&1 < /dev/null &
B=$!
CUDA_VISIBLE_DEVICES=3 setsid nohup bash /tmp/p11_chain_c.sh > logs/p11_g3.log 2>&1 < /dev/null &
C=$!
echo "P11 3-GPU: a pid=$A (GPU0: rn34+regnet), b pid=$B (GPU2: rn50), c pid=$C (GPU3: shuffle)"
echo "launched $(date '+%F %T'); wall ~4.5-6h; markers: /tmp/p11_done_g{0,2,3}"
