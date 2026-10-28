"""生成 CIFAR-10 的 train/test ImageFolder 划分（落机械盘）。

优先下载 HF 的 test parquet；若不可达则从 train 做确定性 holdout 划分。
划分是**确定性**的（固定随机种子），保证多次实验的 train/test 完全一致。
"""
import os
import io
import numpy as np
from PIL import Image
import pandas as pd

ROOT = str(Path(__file__).resolve().parent.parent / "data" / "raw")
OUT = os.path.join(ROOT, 'cifar_full')
NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
         'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs(OUT, exist_ok=True)

train_parquet = os.path.join(ROOT, 'cifar10.parquet')
test_parquet = os.path.join(ROOT, 'cifar10_test.parquet')

df_tr = pd.read_parquet(train_parquet)
print('train rows:', len(df_tr))

if os.path.exists(test_parquet):
    df_te = pd.read_parquet(test_parquet)
    print('test rows:', len(df_te), '(from parquet)')
    mode = 'parquet'
else:
    # 确定性 holdout：从 train 划出 10% 作 test
    rng = np.random.RandomState(1234)
    idx = rng.permutation(len(df_tr))
    n_test = len(df_tr) // 10
    te_idx, tr_idx = idx[:n_test], idx[n_test:]
    df_te = df_tr.iloc[te_idx].reset_index(drop=True)
    df_tr = df_tr.iloc[tr_idx].reset_index(drop=True)
    print('train rows:', len(df_tr), 'holdout test rows:', len(df_te))
    mode = 'holdout'


def dump(df, split):
    d_root = os.path.join(OUT, split)
    os.makedirs(d_root, exist_ok=True)
    lab = df['label'].values
    img = df['img'].values
    for c in range(10):
        idx = np.where(lab == c)[0]
        d = os.path.join(d_root, NAMES[c])
        os.makedirs(d, exist_ok=True)
        for j, i in enumerate(idx):
            p = os.path.join(d, f'{j:05d}.png')
            if not os.path.exists(p):
                Image.open(io.BytesIO(img[i]['bytes'])).convert('RGB').save(p)
    tot = sum(len(os.listdir(os.path.join(d_root, c))) for c in os.listdir(d_root))
    print(f'{split}: {len(os.listdir(d_root))} classes, {tot} images')


dump(df_tr, 'train')
dump(df_te, 'test')
print('cifar_full ready (mode=%s)' % mode)
