#!/usr/bin/env bash
# N4 双臂（ADR-0020 冻结设计）：
#   GPU0 = D 臂：FF+dropout(0.1) × 3 种子（6 run 含 REC 6/7/8 给 N4b）+ P 臂评估（用新存 ckpt，但旧 run 无 ckpt——重跑 rec/ff 0-2 带 ckpt 保存）
#   简化：GPU0 = D 臂 3 run + 重跑 rec/ff s0-2（同协议同种子，产物与 N1″ 同分布且带 ckpt）
#   GPU2 = N4b 新种子：rec/ff × s6/7/8 = 6 run（带 ckpt，供腐蚀评估）
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/n4a.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
# D 臂：FF+dropout（N4 目录）
N4A_TAG=1 CUDA_VISIBLE_DEVICES=0 $PY code/train_hierarchy_n1.py --cfgs ff --seeds 0 1 2 --epochs 100 --feat-dropout 0.1
# P 臂基底：rec/ff s0-2 重跑（n4 前缀，带 ckpt；协议同 N1″ 可比）
N4A_TAG=1 CUDA_VISIBLE_DEVICES=0 $PY code/train_hierarchy_n1.py --cfgs rec ff --seeds 0 1 2 --epochs 100
# P 臂评估
CUDA_VISIBLE_DEVICES=0 $PY code/n4a_probe_decay.py
echo done > /tmp/n4a_done_g0
INNER

cat > /tmp/n4b.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
N4A_TAG=1 CUDA_VISIBLE_DEVICES=2 $PY code/train_hierarchy_n1.py --cfgs rec ff --seeds 6 7 8 --epochs 100
echo done > /tmp/n4b_done_g2
INNER

chmod +x /tmp/n4a.sh /tmp/n4b.sh
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/n4a.sh > logs/n4a_g0.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/n4b.sh > logs/n4b_g2.log 2>&1 < /dev/null &
B=$!
echo "N4: D-arm+P-base pid=$A (GPU0, 9 run + decay eval), N4b pid=$B (GPU2, 6 run)"
echo "launched $(date '+%F %T'); markers: /tmp/n4a_done_g0 /tmp/n4b_done_g2"
