# Two explanations for a moving marker

## 1. Degradation sweep: the input information changes

One network (resnet50, 960 samples), three input conditions. Both the
marker and the probe accuracy fall together, and they fall by nearly the
same proportion.

| condition | effective dim | local smoothness | probe acc | dim vs clean | probe vs clean |
|---|---|---|---|---|---|
| clean | 316 | 0.934 | 0.531 | +0% | +0% |
| occ30 | 287 | 0.910 | 0.438 | -9% | -18% |
| occ70 | 128 | 0.586 | 0.264 | -59% | -50% |
| noise | 271 | 0.899 | 0.309 | -14% | -42% |

The two quantities move together (-59% and -50%). That is
what the information confound predicts and what a genuine degradation of the
representation would also predict. **This comparison cannot tell the two apart.**

Note also what the sweep does not support: the marker shrinks most under 70%
occlusion (a structural loss) and least under noise (an amplitude
perturbation), so it is sensitive to how the information was removed, not
only to how much.

## 2. Recipe sweep: the input information is fixed

42 networks on identical clean inputs, differing only in the training recipe.
If a marker tracked representational quality, it would have to order the
recipes by quality. It does not.

| marker | kind | pooled r | p | pooled rho | interpretation |
|---|---|---|---|---|---|
| `criticality_proxy` | non-trivial | -0.217 | 0.1678 | -0.015 | negative, and driven by the collapsed/healthy split |
| `effective_dim` | non-trivial | -0.483 | 0.0012 | -0.536 | negative, and driven by the collapsed/healthy split |
| `linear_probe_acc` | trivial | +0.993 | 0.0000 | +0.953 | uses test labels; a restatement of the target |
| `local_smoothness` | non-trivial | -0.452 | 0.0027 | -0.338 | negative, and driven by the collapsed/healthy split |
| `silhouette_gap` | trivial | +0.768 | 0.0000 | +0.732 | uses test labels; a restatement of the target |
| `spectral_entropy` | non-trivial | -0.410 | 0.0070 | -0.185 | negative, and driven by the collapsed/healthy split |

### The correlation is a cluster separation, not a gradient

The test accuracy distribution is sharply bimodal: a failed group and a
healthy group, with nothing in between. Within either group the markers carry
no information.

Trained successfully (final train acc >= 0.90): 36 models; collapsed: 6 models.

**By training outcome.** The sign of the pooled correlation is set by where
the collapsed models sit, not by a relationship between the marker and
quality within either group.

| subset | n | `effective_dim` | `local_smoothness` |
|---|---|---|---|
| trained successfully | 36 | r=-0.586 (p=0.000) | r=-0.424 (p=0.010) |
| collapsed | 6 | r=+0.922 (p=0.009) | r=-0.742 (p=0.091) |

**By performance cluster.** The pooled correlation is carried entirely by
the separation between the two performance clusters. Inside either cluster
the marker carries no information about performance.

| cluster | n | `effective_dim` | `local_smoothness` |
|---|---|---|---|
| low cluster (6 recipes) | 18 | r=+0.074 (p=0.770) | r=+0.318 (p=0.199) |
| high cluster (3 recipes) | 9 | r=+nan (p=nan) | r=-0.053 (p=0.891) |

The clusters are defined by recipe, not by a threshold on accuracy, so membership is fixed across seeds. The remaining recipes sit between them.

The undefined coefficient for `effective_dim` in the high cluster is not a computation failure: the marker takes the same value for all nine models there, so the correlation is undefined. That is itself the finding -- in the range where the representation is best, this marker does not vary at all, and so cannot order the models.

**Within a recipe.** Changing only the random seed, the marker barely
moves -- so it measures the recipe, not the individual network's quality.

| marker | residual r (recipe removed) | p |
|---|---|---|
| `effective_dim` | -0.443 | 0.0033 |
| `local_smoothness` | -0.079 | 0.6196 |
| `criticality_proxy` | +0.352 | 0.0223 |
| `spectral_entropy` | -0.521 | 0.0004 |

### A direct counter-example

Two recipes with the same test accuracy but a different marker value:

| recipe | mean test acc | mean local smoothness |
|---|---|---|
| `plain_ls` | 0.7892 | 0.0694 |
| `wd_high` | 0.7897 | 0.1424 |

The accuracies differ by 0.05 percentage points while the
marker differs by 0.0730. **The marker does not resolve a
difference in the quantity it is supposed to track.**

## What this establishes

1. When the input information changes, the marker moves with the probe
   accuracy -- as an arithmetic consequence, not as evidence about the
   representation.
2. When the input information is held fixed, the marker does not track quality
   at all: the pooled correlation is negative and the within-cluster
   correlation collapses.
3. A marker of this family is therefore not a measurement of representational
   quality, and a correlation between it and a performance measure is not
   evidence that the representation is better.

The corollary is the practical one: any claim that such a marker predicts
performance has to survive the recipe sweep, not only the degradation sweep.
That test is cheap, and it is the one that disqualifies the marker.
