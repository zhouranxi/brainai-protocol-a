#!/usr/bin/env bash
# A0 门控探针（ADR-0024）：仅用 GPU2/GPU3（用户约束）。
#   GPU2 = bfm×3 + ff×3（6 run，双前向较贵）
#   GPU3 = rec×3（3 run，最轻——与 llama-server 共卡，显存 ~2GB）
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/a0_g2.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
~/venvs/dl/bin/python - <<'PYE'
import sys; sys.path.insert(0, 'code')
import train_a0, torch
for s in (0, 1, 2):
    train_a0.train_one('bfm', s, torch.device('cuda:0'))
for s in (0, 1, 2):
    train_a0.train_one('ff', s, torch.device('cuda:0'))
PYE
echo done > /tmp/a0_done_g2
INNER

cat > /tmp/a0_g3.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
~/venvs/dl/bin/python - <<'PYE'
import sys; sys.path.insert(0, 'code')
import train_a0, torch
for s in (0, 1, 2):
    train_a0.train_one('rec', s, torch.device('cuda:0'))
PYE
echo done > /tmp/a0_done_g3
INNER

chmod +x /tmp/a0_g2.sh /tmp/a0_g3.sh
mkdir -p logs
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/a0_g2.sh > logs/a0_g2.log 2>&1 < /dev/null &
A=$!
CUDA_VISIBLE_DEVICES=3 setsid nohup bash /tmp/a0_g3.sh > logs/a0_g3.log 2>&1 < /dev/null &
B=$!
echo "A0: g2(bfm+ff) pid=$A, g3(rec) pid=$B"
echo "launched $(date '+%F %T'); markers: /tmp/a0_done_g{2,3}"
