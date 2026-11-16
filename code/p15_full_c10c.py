#!/usr/bin/env python3
"""P15 (ADR-0021 P15 candidate, user-approved): full-batch c10c re-measure.

Every p*_* run dir under v2_p4_forcing gets its final whiten
checkpoint evaluated on CIFAR-10-C (19x5). Resume-safe (skips
points already in the output JSON).
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
os.environ.setdefault('BRAINAI_DATA_ROOT', '/mnt/mydisk/xinguan-brainai/data')
import numpy as np
import torch
from torchvision import transforms
from train_forcing_p4 import CIFAR10_MEAN, CIFAR10_STD, DATA, build_model, set_seed

RUNS = Path('/mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing')
OUT = RUNS / 'p15_full_c10c.json'
root = os.path.join(DATA, 'CIFAR-10-C')
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
labels_np = np.load(os.path.join(root, 'labels.npy'))
corruptions = sorted(p.stem for p in Path(root).glob('*.npy') if p.stem != 'labels')
imgs_cache = {c: np.load(os.path.join(root, f'{c}.npy')) for c in corruptions}

def eval_np(model, bs=512):
    model.eval(); corr = tot = 0
    with torch.no_grad():
        for c in corruptions:
            imgs = imgs_cache[c]; ys = torch.as_tensor(labels_np[:len(imgs)]).long()
            for i in range(0, len(imgs), bs):
                x = torch.stack([tf(im) for im in imgs[i:i+bs]]).cuda()
                yb = ys[i:i+bs].cuda()
                corr += (model(x).argmax(1) == yb).sum().item(); tot += len(yb)
    return corr / tot

rows = []
done = set()
if OUT.exists():
    rows = json.load(open(OUT))['rows']
    done = {r['run'] for r in rows}
    print(f'[resume] {len(done)} done', flush=True)

dirs = sorted(d.name for d in RUNS.iterdir()
              if d.is_dir() and d.name[0] == 'p' and '_' in d.name
              and not d.name.startswith('p14'))
device = torch.device('cuda:0')
for k, dname in enumerate(dirs):
    if dname in done: continue
    cks = sorted((RUNS / dname).glob('*whiten*.pt'))
    if not cks: continue
    ck = cks[-1]
    # 架构按目录名解析；p11 批（架构异质 + ADR-0017 判失败）跳过
    if dname.startswith('p11'):
        continue
    arch = 'resnet18'
    if 'regnet' in dname: arch = 'regnet_y_400mf'
    elif 'shufflenet' in dname: arch = 'shufflenet_v2_x1_0'
    elif 'resnet50' in dname: arch = 'resnet50'
    elif 'resnet34' in dname: arch = 'resnet34'
    set_seed(123)
    model = build_model(arch).to(device)
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    acc = round(eval_np(model), 4)
    rows.append({'run': dname, 'ckpt': ck.name, 'c10c_mean': acc})
    json.dump({'rows': rows}, open(OUT, 'w'), indent=1)
    print(f'[{k+1}/{len(dirs)}] {dname}: {acc:.4f}', flush=True)
    del model; torch.cuda.empty_cache()
print(f'ALL DONE: {len(rows)} runs -> {OUT}', flush=True)
