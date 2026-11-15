#!/usr/bin/env bash
# ADR-0019 N1：3 配置 × 3 种子 = 9 run，GPU0/GPU2 双链，墙钟 ~2h。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/n1_a.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
$PY code/train_hierarchy_n1.py --cfgs ff rec hfb --seeds 0 1 2 --epochs 150
echo done > /tmp/n1p_done_g0
INNER

cat > /tmp/n1_b.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
# B 链按种子切分（A 跑完 9 个中的大部分，B 补齐避免串行瓶颈：B 跑 seed2 的三个配置）
$PY code/train_hierarchy_n1.py --cfgs ff rec hfb --seeds 2 --epochs 150
echo done > /tmp/n1p_done_g2
INNER

chmod +x /tmp/n1_a.sh /tmp/n1_b.sh
mkdir -p logs
# 注意：skip 逻辑保证两链不重复跑（JSON 已存在则跳过）
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/n1_a.sh > logs/n1p_g0.log 2>&1 < /dev/null &
A=$!
sleep 5
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/n1_b.sh > logs/n1p_g2.log 2>&1 < /dev/null &
B=$!
echo "N1' chainA pid=$A (GPU0: all 9), chainB pid=$B (GPU2: seed2 first, skip-if-done)"
echo "launched $(date '+%F %T'); markers: /tmp/n1_done_g{0,2}"
