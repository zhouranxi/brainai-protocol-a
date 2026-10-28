# Protocol A: When Do Geometric Markers Predict Generalization?

**Code, data, and analysis for** *"When Do Geometric Markers Predict Generalization? A Controlled Analysis of Task-Relevance and Shift Type"*

This repository contains the complete four-condition control protocol, the analysis code, the
released per-model measurements it operates on, the independently trained models, and every
number reported in the paper — together with a script that re-derives each reported number
from the raw data.

> **One-command verification.** `python code/audit_claims.py` recomputes **133 headline numbers**
> directly from the raw data using the same protocol as the analysis scripts and checks each
> against the value written in the paper. It is the fastest way to confirm that this repository
> and the manuscript agree — no GPU, no downloads, about a minute.
>
> The same check runs automatically on every push (`.github/workflows/verify.yml`), so the
> repository's headline claim — *every reported number is reproducible from the released
> artifacts* — is continuously machine-verified rather than asserted.

---

## 1. What this archive contains

| Path | Contents |
|---|---|
| `code/` | All analysis and training scripts (17 active) |
| `code/archive/` | Earlier exploratory scripts kept for provenance (22 files, not part of the final pipeline) |
| `data/official/` | The two released per-model datasets from the original study (unmodified) |
| `data/self_trained/` | Per-model measurements for the 80 models we trained (ResNet-18, ShuffleNetV2, RegNetY) |
| `results/` | Every table, report, and JSON produced by the pipeline |
| `figures/` | Figures 2–6 (PNG 300 dpi + PDF vector) |
| `paper/` | Manuscript source (TMLR LaTeX), compiled PDF, and internal drafts |
| `DATA_AVAILABILITY.md` | Where to get the 80 model checkpoints (Zenodo) and how to verify this repo |

**Model checkpoints (2.4 GB) are not in this repository** — they are distributed
as a Zenodo archive; see [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

## 2. The protocol in one page

For a candidate predictor $x$ (a geometric marker) and a target $y$ (OOD accuracy), an
association is reported as evidence only if **all four** conditions hold:

| | Condition | What it controls |
|---|---|---|
| **C1** | ID matching | Restrict to models with near-identical ID accuracy ($\pm b$ pp) — the baseline's own variation must not drive the comparison |
| **C2** | Group residualization | Regress out group main effects (architecture / learning-rate family) via dummy-variable OLS |
| **C3** | Dual correlation statistics | Pearson $r$ and Spearman $\rho$ must agree in sign and $|\rho| \ge 0.5$ |
| **C4** | Cross-group sign consistency | The within-group correlation must have the same sign in every group with $n_g \ge 5$ |

The protocol is applied **symmetrically**: the same four conditions are imposed on the proposed
proxy *and* on the ID-accuracy baseline. Applying C4 to the baseline is what reveals its sign
instability (§4.7 of the paper).

## 3. Data provenance

### 3.1 Official releases (used unmodified)

| File | Models | Source |
|---|---|---|
| `data/official/df_fig3.pkl` | 224 | `ood-generalization-geometry` repository, `data/df_fig3.pkl` |
| `data/official/df_fig15-20.pkl` | 80 × 6 corruption scenarios | same repository, `data/df_fig15-20.pkl` |

Verified: MD5 `8537fb309647e0cd8f981a66e2c2d98a` (`df_fig3.pkl`) and
`75c5ed80c2989007d8df473fa559708e` (`df_fig15-20.pkl`), identical to the upstream repository
copies. The two releases share **no models** (their architecture/hyperparameter/seed keys
intersect empty), and in the second release the `acc_lp_ood` column is *identical* across all
six corruption scenarios (pairwise correlation 1.0000, maximum absolute difference 0.000),
i.e. it is a scenario-independent class-level transfer measure.

### 3.2 Self-trained models (our own)

80 models trained from scratch on CIFAR-10:

| Architecture | Grid | Seeds | Models |
|---|---|---|---|
| ResNet-18 | 4 learning rates × 4 weight decays | 3 | 48 |
| ShuffleNetV2-x1.0 | reduced grid | 2 | 16 |
| RegNetY-400MF | reduced grid | 2 | 16 |

200 epochs, cosine schedule, batch size 128. Evaluated with publicly computable markers on
two class-level shifts (CIFAR-100 and Tiny-ImageNet linear probes, frozen features, new head)
and 19 CIFAR-10-C corruption scenarios (severity 5).

**One model diverged** (`shufflenet_v2_x1_0_lr0.5_wd0.0002_s1`, ID accuracy 0.10) and is
excluded from every analysis; it is retained in `checkpoints/` because the exclusion is part of
the reported procedure. One learning-rate family in the 48-model ResNet-18 collection
(`lr=1.0`) does not reach a comparable OOD level; it is retained in the headline numbers and
a converged-family sensitivity analysis excluding it is reported in §4.7.

### 3.3 GLUE implementation

`code/glue_impl.py` is our own implementation of the GLUE dimension, following the published
Algorithm 1 and the anchor-point solver of Chung, Lee & Sompolinsky (2018). The reference
implementation was not available to us, and the paper's appendix specifies two steps
(per-manifold-pair analysis, Gaussianization preprocessing) that ours does not cover. It is
therefore used **only for relative comparisons within our own model family** (§4.7b), never
compared to released absolute values.

## 4. Reproducing the paper

### 4.1 Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Reference environment: Python 3.12.3, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130,
CUDA 13.0, 4 × NVIDIA RTX 3090 (24 GB). CPU-only execution is possible for the analysis
scripts (no GPU needed) but not for training or measurement.

### 4.2 Verify the archive (no GPU, ~1 minute)

```bash
python code/audit_claims.py
# -> "claims: 133  pass: 133  FAIL: 0"
```

### 4.3 Regenerate the figures (no GPU, ~2 minutes)

```bash
python code/make_figures.py --outdir figures
```

### 4.4 Re-run the analyses (no GPU, ~10 minutes)

```bash
python analyze_fig3.py            # official 224-model release, Protocol A  -> results/protocol_a_fig3_report.md
python analyze_independent.py      # self-trained 48 models, two class-level shifts
python analyze_multiarch.py        # 79 models across three architectures
python bootstrap_ci.py             # stratified bootstrap CIs, 2000 replicates
python analyze_ind48_robust.py     # C4 per-family check + converged-family sensitivity
python analyze_corruption_dual.py  # shift-type decomposition under two instruments
```

### 4.5 Download the evaluation data

```bash
bash code/fetch_ood.sh        # CIFAR-100, CIFAR-10-C
bash code/fetch_cifar_full.py # CIFAR-10/100 parquet mirrors
```

### 4.6 Train from scratch (GPU, ~14 h on one RTX 3090)

```bash
python code/train_grid.py --arch resnet18 --epochs 200 --seeds 0 1 2 \
    --lr-grid 0.5 0.25 0.125 0.0625 --wd-grid 0.0001 0.0002 0.0005 0.001 \
    --outdir checkpoints/resnet18
```

### 4.7 Measure markers and OOD accuracy (GPU, ~2.5 min per model)

```bash
python code/measure_independent.py --probe-epochs 20 --out runs
python code/measure_mod.py        --probe-epochs 20 --out runs   # new architectures
```

The shipped checkpoints are already measured; the per-model measurements are in
`data/self_trained/` (`ind_validation_results.json`, `mod_arch_results.json`,
`glue_own48_results.json`), so the analysis scripts run without a GPU.

## 5. Where each paper number comes from

| Paper location | Produced by | Output file |
|---|---|---|
| §4.2 full-sample landscape | `code/analyze_fig3.py` | `results/protocol_a_fig3_report.md` (F1) |
| §4.3 Table 2 (primary) | `code/analyze_fig3.py` | `results/protocol_a_fig3_report.md` (F2) |
| §4.3 Table 3 (Simpson) | `code/analyze_fig3.py` | `results/protocol_a_fig3_report.md` (F2) |
| §4.4 band sensitivity | `code/analyze_fig3.py` | `results/protocol_a_fig3_report.md` (F4) |
| §4.5 bootstrap CIs | `code/bootstrap_ci.py` | `results/bootstrap_ci_report.md` |
| §4.6 shift-type decomposition | `code/analyze_corruption_dual.py` | `results/corruption_dual_report.md` |
| §4.6 official 6-scenario table | `code/analyze_fig3.py` | `results/r1_final_report.md` (T5) |
| §4.7 self-trained replication | `code/analyze_independent.py` | `results/ind_validation_report.md` |
| §4.7 C4 per-family + sensitivity | `code/analyze_ind48_robust.py` | `results/ind48_robust_report.md` |
| §4.7a cross-architecture | `code/analyze_multiarch.py` | `results/multiarch_report.md` |
| §4.7b cross-implementation | `code/glue_own48.py` | `runs/glue_own48_results.json` |
| §4.7c orthogonality | `code/analyze_ind48_robust.py` | `results/ind48_robust_results.json` |
| Figures 2–6 | `code/make_figures.py` | `figures/fig{2..6}.{png,pdf}` |
| All headline numbers | `code/audit_claims.py` | `results/claims_audit_report.md` |

## 6. Citation

If you use this archive, please cite the paper. A `CITATION.cff` file is included for
automated citation; add the final venue and DOI here once the paper is published.

## 7. License

Code and data in this archive are released under the MIT License (see `LICENSE`). The two
official datasets in `data/official/` are the work of Chou, Kirsanov, Yang & Chung and are
redistributed unmodified from the MIT-licensed `ood-generalization-geometry` repository; they
remain under their original authors' terms. CIFAR-10/CIFAR-100/CIFAR-10-C/Tiny-ImageNet are
used under their respective licenses and are **not** redistributed here (download scripts are
provided instead).
