"""下载 R1 所需的 OOD 数据集（CIFAR-100 / CIFAR-10C），落机械盘。

对应 ICLR 2026 论文的两类 OOD：
  - CIFAR-100：class-level shift（类别不重合），论文主实验用
  - CIFAR-10C ：corruption shift（标签空间不变），论文的对照 —— 在此场景下
                ID accuracy 反而是最强预测器，几何标记失效
两者一起才能刻画「几何标记的预测力是否分布偏移专属」。

数据源：HF mirror（直连 GitHub/toronto 慢）
"""
import os
import io
import tarfile
import urllib.request

import numpy as np
from PIL import Image

ROOT = str(Path(__file__).resolve().parent.parent / "data" / "raw")
os.makedirs(ROOT, exist_ok=True)

MIRROR = 'https://hf-mirror.com'


def http_get(url, dest, timeout=300):
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'research'})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, 'wb') as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (50 << 20) < (1 << 20):
                print(f"    {total/1e6:.0f} MB")
    print(f"  -> {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")


# ---------------------------------------------------------------- CIFAR-100
def fetch_cifar100():
    out = os.path.join(ROOT, 'cifar100')
    if os.path.isdir(os.path.join(out, 'test')):
        print("cifar100 already present")
        return
    print("=== CIFAR-100 ===")
    base = f"{MIRROR}/datasets/uoft-cs/cifar100/resolve/main/cifar100"
    for split in ['train', 'test']:
        dest = os.path.join(ROOT, f"cifar100_{split}.parquet")
        if not os.path.exists(dest):
            http_get(f"{base}/{split}-00000-of-00001.parquet", dest)
        import pandas as pd
        df = pd.read_parquet(dest)
        print(f"  {split}: rows={len(df)} cols={list(df.columns)}")
        names = [f'c{i:02d}' for i in range(100)]
        d_root = os.path.join(out, split)
        os.makedirs(d_root, exist_ok=True)
        lab = df['fine_label'].values if 'fine_label' in df.columns else df['label'].values
        img = df['img'].values
        for c in range(100):
            idx = np.where(lab == c)[0]
            d = os.path.join(d_root, names[c])
            os.makedirs(d, exist_ok=True)
            for j, i in enumerate(idx):
                p = os.path.join(d, f'{j:05d}.png')
                if not os.path.exists(p):
                    Image.open(io.BytesIO(img[i]['bytes'])).convert('RGB').save(p)
        print(f"  {split}: {len(os.listdir(d_root))} classes, "
              f"{sum(len(os.listdir(os.path.join(d_root, c))) for c in os.listdir(d_root))} images")


# ---------------------------------------------------------------- CIFAR-10C
def fetch_cifar10c():
    out = os.path.join(ROOT, 'cifar10c')
    if os.path.isdir(os.path.join(out, 'test')):
        print("cifar10c already present")
        return
    print("=== CIFAR-10C ===")
    tarball = os.path.join(ROOT, 'cifar10c.tar.gz')
    if not os.path.exists(tarball):
        # HF 上有打包版本；先试 GitHub release（可能不通），再回退 HF
        for url in [
            f"{MIRROR}/datasets/torch-uncertainty/CIFAR-C/resolve/main/CIFAR-10-C.tar",
            "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1",
        ]:
            try:
                print(f"  try {url}")
                http_get(url, tarball, timeout=600)
                break
            except Exception as e:
                print(f"  failed: {e}")
                continue
    if not os.path.exists(tarball) or os.path.getsize(tarball) < 1e6:
        print("  !! cifar10c download failed — will skip (corruption control arm)")
        return
    import pandas as pd
    with tarfile.open(tarball) as tf:
        tf.extractall(ROOT)
    src = os.path.join(ROOT, 'CIFAR-10-C')
    if not os.path.isdir(src):
        src = os.path.join(ROOT, 'cifar-10-c')
    if not os.path.isdir(src):
        # tar 内可能是 CIFAR-10-C/ 或直接在根
        import glob
        cands = glob.glob(os.path.join(ROOT, '*CIFAR*')) + glob.glob(os.path.join(ROOT, '*cifar*'))
        cands = [c for c in cands if os.path.isdir(c)]
        if cands:
            src = cands[0]
    if not os.path.isdir(src):
        print("  !! extracted dir not found, skip")
        return
    # 取 15 个 corruption 中规模适中的若干个做测试集
    d_root = os.path.join(out, 'test')
    os.makedirs(d_root, exist_ok=True)
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']
    # 取 severity=5（最严重），全部 15 种 corruption；每类每 corruption 取 100 张
    # 论文用的是 CIFAR-10C 全集；这里取最严重档以放大分布偏移信号
    SEV = 5
    all_corr = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
                'gaussian_blur', 'defocus_blur', 'glass_blur', 'motion_blur',
                'zoom_blur', 'fog', 'frost', 'snow', 'contrast',
                'elastic_transform', 'pixelate', 'jpeg_compression',
                'brightness', 'saturate', 'spatter']
    labels = np.load(os.path.join(src, 'labels.npy'))
    lab_sev = labels[(SEV - 1) * 10000: SEV * 10000]
    for corr in all_corr:
        f = os.path.join(src, f'{corr}.npy')
        if not os.path.exists(f):
            continue
        d = os.path.join(d_root, corr)
        if os.path.isdir(d):
            continue
        arr = np.load(f)[(SEV - 1) * 10000: SEV * 10000]
        os.makedirs(d, exist_ok=True)
        for c in range(10):
            idx = np.where(lab_sev == c)[0][:100]
            cd = os.path.join(d, names[c])
            os.makedirs(cd, exist_ok=True)
            for j, i in enumerate(idx):
                Image.fromarray(arr[i]).convert('RGB').save(os.path.join(cd, f'{j:04d}.png'))
        print(f"  {corr}: {len(os.listdir(d))} classes x 100")


if __name__ == '__main__':
    fetch_cifar100()
    fetch_cifar10c()
    print("DONE")
