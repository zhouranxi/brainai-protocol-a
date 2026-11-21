#!/usr/bin/env bash
# PC-A0 (ADR-0029): GPU2 = beta sweep, GPU3 shares seeds 2 / shorter arms.
# Total 9 runs (~0.5 GPU.h each incl. eval) -> ~2h on two cards.
set -e
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data

cat > /tmp/pc_g2.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
~/venvs/dl/bin/python code/train_pc_a0.py --betas 0.0,0.1 --seeds 0,1,2
~/venvs/dl/bin/python code/train_pc_a0.py --betas 1.0 --seeds 0
echo done > /tmp/pc_done_g2
INNER

cat > /tmp/pc_g3.sh <<'INNER'
#!/usr/bin/env bash
set -e
cd /mnt/mydisk/xinguan-brainai
export BRAINAI_DATA_ROOT=/mnt/mydisk/xinguan-brainai/data
~/venvs/dl/bin/python code/train_pc_a0.py --betas 1.0 --seeds 1,2
echo done > /tmp/pc_done_g3
INNER

chmod +x /tmp/pc_g2.sh /tmp/pc_g3.sh
mkdir -p logs
CUDA_VISIBLE_DEVICES=2 setsid nohup bash /tmp/pc_g2.sh > logs/pc_g2.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 setsid nohup bash /tmp/pc_g3.sh > logs/pc_g3.log 2>&1 < /dev/null &
echo "PC-A0: g2(b0/b0.1 all + b1.0 s0) + g3(b1.0 s1,s2) launched $(date '+%F %T')"
