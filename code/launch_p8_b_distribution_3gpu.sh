#!/usr/bin/env bash
# P8 三卡并行版（ADR-0017 预注册不变，仅调度改进）：
#   GPU0 -> s0 4 run（lam 0.03/0.1/1.0/3.0，run-seed 2）
#   GPU2 -> s1 5 run（全剂量轴，run-seed 2）
#   GPU3 -> s2 5 run（全剂量轴，run-seed 2）
# 墙钟 ~2.8h（原单卡 7.7h）。协议与 ADR-0016 附录二十三~三十五逐字一致。
# 完成标记：三条链各自写 /tmp/p8_done_g{0,2,3}，全部出现即收批。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

make_chain() {  # $1=seed-tag(s0/s1/s2)  $2=GPU  $3=lam 列表  $4=marker
  cat > /tmp/p8_chain_$1.sh <<INNER
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
for LAM in $3; do
  \$PY code/train_forcing_p4.py --arm full --epochs 200 --skip-base \\
    --out runs/v2_p4_forcing/p8_$1_seed2_lam\${LAM} \\
    --seed 2 --kinds whiten --lambda \${LAM} \\
    --models resnet18_lr0.125_wd0.0001_$1
done
echo done > $4
INNER
  chmod +x /tmp/p8_chain_$1.sh
}

make_chain s0 0 "0.03 0.1 1.0 3.0" /tmp/p8_done_g0
make_chain s1 2 "0.03 0.1 0.3 1.0 3.0" /tmp/p8_done_g2
make_chain s2 3 "0.03 0.1 0.3 1.0 3.0" /tmp/p8_done_g3

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p8_chain_s0.sh > logs/p8_g0_s0.log 2>&1 < /dev/null &
P0=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p8_chain_s1.sh > logs/p8_g2_s1.log 2>&1 < /dev/null &
P2=$!
CUDA_VISIBLE_DEVICES=3 setsid nohup bash /tmp/p8_chain_s2.sh > logs/p8_g3_s2.log 2>&1 < /dev/null &
P3=$!
echo "P8 3-GPU launch: s0 pid=$P0 (GPU0), s1 pid=$P2 (GPU2), s2 pid=$P3 (GPU3)"
echo "launched $(date '+%F %T'); wall clock ~2.8h; markers: /tmp/p8_done_g{0,2,3}"
