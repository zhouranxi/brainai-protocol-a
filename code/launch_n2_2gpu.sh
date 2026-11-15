#!/usr/bin/env bash
# N2 双臂并行（ADR-0019 第八节冻结设计）：
#   GPU0 = N2a 缩放扫描：hfb × α{0.1,0.3,1.0} × seed{0,1,2} = 9 run
#   GPU2 = N2b REC-OOD：ff/rec × seed{3,4,5} = 6 run（与 N1″ 合并成 6 种子）
# 注意 N2b 的 n1p 目录名复用（skip 逻辑防重复；seed 3-5 是新的）。
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/n2a.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
for A in 0.1 0.3 1.0; do
  $PY code/train_hierarchy_n1.py --cfgs hfb --seeds 0 1 2 --epochs 100 --fb-scale $A
done
echo done > /tmp/n2a_done_g0
INNER

cat > /tmp/n2b.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
$PY code/train_hierarchy_n1.py --cfgs ff rec --seeds 3 4 5 --epochs 100
echo done > /tmp/n2b_done_g2
INNER

chmod +x /tmp/n2a.sh /tmp/n2b.sh
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup bash /tmp/n2a.sh > logs/n2a_g0.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/n2b.sh > logs/n2b_g2.log 2>&1 < /dev/null &
B=$!
echo "N2: a-scale-sweep pid=$A (GPU0, 9 runs), b-rec-ood pid=$B (GPU2, 6 runs)"
echo "launched $(date '+%F %T'); markers: /tmp/n2a_done_g0 /tmp/n2b_done_g2"
