# Data and Model Availability

This repository ships the **code**, the **analysis outputs**, and the **released
per-model measurements**. Model checkpoints are distributed separately.

## What is in this repository

| Path | Size | Contents |
|---|---|---|
| `code/` | — | 17 analysis/training scripts (+ 22 archived exploratory scripts) |
| `data/official/` | 370 KB | The two released per-model datasets from the original study, **unmodified** |
| `data/self_trained/` | 260 KB | Per-model measurements for the 80 models we trained (JSON) |
| `results/` | 550 KB | Every table, report, and JSON produced by the pipeline |
| `figures/` | 1.1 MB | Figures 2–6 (PNG 300 dpi + PDF vector) |
| `paper/` | 1.3 MB | Manuscript source (TMLR LaTeX), compiled PDF, and internal drafts |

Total: **100 files, ~4.3 MB.** Everything is plain text or PNG/PDF — no build step
required to read it.

## Model checkpoints (2.4 GB) — not in this repository

The 80 trained checkpoints (48 ResNet-18 + 16 ShuffleNetV2-x1.0 + 16 RegNetY-400MF)
are **2.4 GB**, which is too large for a Git repository even with Git LFS. They are
distributed as a **Zenodo archive**, with a versioned DOI:

> **Zenodo record:** `10.5281/zenodo.23728884` (version DOI)
> **Concept DOI:** `10.5281/zenodo.23728883`
> **Record page:** `https://zenodo.org/records/23728884`
> **Direct download:** `https://zenodo.org/records/23728884/files/`

> The concept DOI is stable across versions, so it is the one to cite; the version
> DOI points at this exact release.

The archive contains two tarballs:

| File | Contents |
|---|---|
| `checkpoints_resnet18.tar` | 48 ResNet-18 checkpoints (our 48-model collection) |
| `checkpoints_newarch.tar` | 32 checkpoints (16 ShuffleNetV2 + 16 RegNetY) |

One model diverged during training (`shufflenet_v2_x1_0_lr0.5_wd0.0002_s1`,
ID accuracy 0.10) and is excluded from every analysis. It is retained in the
archive because the exclusion is part of the reported procedure.

After downloading, unpack with:

```bash
tar -xf checkpoints_resnet18.tar -C checkpoints/resnet18
tar -xf checkpoints_newarch.tar -C checkpoints/newarch
```

## Verifying this repository without any downloads

The measurements needed to reproduce every number in the paper are already in
`data/`. The self-verification script runs on CPU in about a minute:

```bash
pip install -r requirements.txt
bash run_all.sh verify
# expected: claims: 133  pass: 133  FAIL: 0
```

It re-derives **133 reported numbers** directly from the shipped data using the
same protocol as the analysis scripts and checks each against the value written
in the manuscript. This is the intended entry point for a reader who wants to
check the work rather than rerun the training.

## Third-party data

`data/official/` contains the per-model measurement releases of:

> Chou, C.-N., Kirsanov, A., Yang, Y.-Y., & Chung, S. (2026).
> *Diagnosing Generalization Failures from Representational Geometry Markers.* ICLR 2026.
> https://github.com/chung-neuroai-lab/ood-generalization-geometry

They are redistributed **unmodified** under that repository's MIT license and
remain the copyright of their original authors. MD5 checksums:

| File | MD5 |
|---|---|
| `df_fig3.pkl` | `8537fb309647e0cd8f981a66e2c2d98a` |
| `df_fig15-20.pkl` | `75c5ed80c2989007d8df473fa559708e` |

CIFAR-10, CIFAR-100, CIFAR-10-C and Tiny-ImageNet are **not** redistributed here.
Download scripts are provided in `code/fetch_ood.sh` and `code/fetch_cifar_full.py`;
each dataset remains under its own license.
