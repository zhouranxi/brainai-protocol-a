#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分步定位：glue_dimension 单测（5000 样本小样）+ base 臂各阶段计时。"""
import sys
import time

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/code")
import numpy as np

t0 = time.time()
print("[1] importing train_forcing_p4...", flush=True)
from train_forcing_p4 import glue_dimension  # noqa: E402

print("    import done", round(time.time() - t0, 1), "s", flush=True)

rng = np.random.default_rng(0)
X = rng.standard_normal((5000, 512)).astype(np.float32)
y = np.repeat(np.arange(10), 500)

print("[2] glue_dimension n_sub=100...", flush=True)
t1 = time.time()
g = glue_dimension(X, y, n_sub=100, seed=0)
print("    result:", g, round(time.time() - t1, 1), "s", flush=True)

print("[3] glue_dimension n_sub=500 (real endpoint)...", flush=True)
t2 = time.time()
g2 = glue_dimension(X, y, n_sub=500, seed=0)
print("    result:", g2, round(time.time() - t2, 1), "s", flush=True)
print("STEP-OK total", round(time.time() - t0, 1), "s", flush=True)
