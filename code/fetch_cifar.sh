#!/bin/bash
# CIFAR-10 下载与转换（数据一律落在机械盘 /mnt/mydisk）
set -e
ROOT=/mnt/mydisk/xinguan-brainai/data
mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d smoke ]; then
  echo '--- downloading cifar10 via hf-mirror ---'
  curl -sL --retry 3 -o cifar10.parquet \
    https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
  ls -la cifar10.parquet

  echo '--- converting parquet -> ImageFolder ---'
  ~/venvs/dl/bin/python - << 'PY'
import os, io, numpy as np
from PIL import Image
import pandas as pd

root = '/mnt/mydisk/xinguan-brainai/data'
out  = os.path.join(root, 'smoke')
os.makedirs(out, exist_ok=True)

df = pd.read_parquet(os.path.join(root, 'cifar10.parquet'))
print('columns:', list(df.columns), 'rows:', len(df))

names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
lab = df['label'].values
img = df['img'].values

rng = np.random.RandomState(0)
for c in range(10):
    idx = np.where(lab == c)[0]
    idx = rng.permutation(idx)[:120]
    d = os.path.join(out, names[c])
    os.makedirs(d, exist_ok=True)
    for j, i in enumerate(idx):
        b = img[i]['bytes']
        Image.open(io.BytesIO(b)).convert('RGB').save(os.path.join(d, f'{j:04d}.png'))

tot = sum(len(os.listdir(os.path.join(out, c))) for c in os.listdir(out))
print('classes:', len(os.listdir(out)), 'total:', tot)
PY
fi

echo 'CIFAR_READY'
du -sh "$ROOT"
