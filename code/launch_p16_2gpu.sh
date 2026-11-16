#!/usr/bin/env bash
# P16 瓶颈批（ADR-0022 冻结设计）：
#   GPU0 = B1 mask + B2 width（18 run）
#   GPU2 = B3 data（9 run）
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/p16_a.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
$PY code/train_hierarchy_p16.py --bottlenecks mask width --cfgs ff rec hfb --seeds 0 1 2 --epochs 100
echo done > /tmp/p16_done_g0
INNER

cat > /tmp/p16_b.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
$PY code/train_hierarchy_p16.py --bottlenecks data --cfgs ff rec hfb --seeds 0 1 2 --epochs 100
echo done > /tmp/p16_done_g2
INNER

chmod +x /tmp/p16_a.sh /tmp/p16_b.sh
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/p16_a.sh > logs/p16_g0.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/p16_b.sh > logs/p16_g2.log 2>&1 < /dev/null &
B=$!
echo "P16: a(mask+width) pid=$A (GPU0, 18 runs), b(data) pid=$B (GPU2, 9 runs)"
echo "launched $(date '+%F %T'); markers: /tmp/p16_done_g{0,2}"
