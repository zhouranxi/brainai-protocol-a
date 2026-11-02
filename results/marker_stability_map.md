# Marker stability map

Protocol: ID matching around the mean of `acc_test` at +/-1 pp, then
architecture residualisation. Three cells: official implementation on the official
224-model release, official implementation on the official 80-model release, and our
own implementation on our own 48 models (CIFAR-100 linear probe as the OOD target).

## The map

| Marker | kind | official 224 | official 80 | our own 48 | classification |
|---|---|---|---|---|---|
| `dimension` | GLUE | r=+0.925 rho=+0.847 (n=106) | r=+0.737 rho=+0.803 (n=25) | r=+0.919 rho=+0.868 (n=32) | **robust** |
| `radius` | GLUE | r=-0.584 rho=-0.639 (n=106) | r=+0.153 rho=+0.322 (n=25) | r=+0.602 rho=+0.673 (n=32) | **SIGN FLIPS (Pearson)** |
| `utility` | GLUE | r=+0.901 rho=+0.769 (n=106) | r=+0.808 rho=+0.896 (n=25) | r=+0.342 rho=+0.135 (n=32) | **consistent sign, variable magnitude** |
| `capacity` | GLUE | r=-0.429 rho=-0.331 (n=106) | r=-0.080 rho=-0.200 (n=25) | n<8 | **consistent sign, weak in at least one cell** |
| `center alignment` | GLUE | r=+0.483 rho=+0.400 (n=106) | r=+0.581 rho=+0.642 (n=25) | n<8 | **robust** |
| `participation ratio (per class)` | generic | r=+0.659 rho=+0.503 (n=106) | r=+0.592 rho=+0.735 (n=25) | r=+0.947 rho=+0.922 (n=32) | **robust** |
| `participation ratio (global)` | generic | r=+0.877 rho=+0.700 (n=106) | r=+0.734 rho=+0.878 (n=25) | r=+0.895 rho=+0.901 (n=32) | **robust** |
| `numerical rank (per class)` | generic | r=+0.436 rho=+0.570 (n=106) | n<8 | r=+0.972 rho=+0.970 (n=32) | **consistent sign, variable magnitude** |
| `numerical rank (global)` | generic | r=+0.556 rho=+0.633 (n=106) | n<8 | n<8 | **insufficient data** |
| `sparsity` | generic | r=+0.268 rho=+0.584 (n=106) | r=+0.390 rho=+0.432 (n=25) | n<8 | **consistent sign, weak in at least one cell** |
| `mean distance` | generic | r=+0.124 rho=+0.296 (n=106) | r=+0.485 rho=+0.607 (n=25) | n<8 | **consistent sign, weak in at least one cell** |
| `mean angle` | generic | r=+0.124 rho=-0.121 (n=106) | r=-0.102 rho=-0.152 (n=25) | n<8 | **SIGN FLIPS (Pearson)** |
| `mean covariance` | generic | r=+0.035 rho=+0.444 (n=106) | r=+0.405 rho=+0.561 (n=25) | n<8 | **consistent sign, weak in at least one cell** |
| `NC1` | generic | r=-0.145 rho=-0.146 (n=106) | n<8 | r=-0.178 rho=+0.097 (n=32) | **SIGN FLIPS (Spearman)** |
| `local smoothness` | ours only | n<8 | n<8 | r=+0.931 rho=+0.949 (n=32) | **insufficient data** |
| `effective dim (PCA)` | ours only | n<8 | n<8 | r=+0.856 rho=+0.936 (n=32) | **insufficient data** |
| `class separability` | ours only | n<8 | n<8 | r=-0.832 rho=-0.799 (n=32) | **insufficient data** |

## Band sensitivity (Pearson r)

### `dimension` (robust)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.925 | +0.737 | +0.919 |
| +/-2 pp | +0.911 | +0.713 | +0.903 |
| +/-3 pp | +0.906 | +0.751 | +0.901 |
| +/-5 pp | +0.904 | +0.776 | +0.908 |

### `radius` (SIGN FLIPS (Pearson))

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | -0.584 | +0.153 | +0.602 |
| +/-2 pp | -0.796 | +0.025 | +0.492 |
| +/-3 pp | -0.790 | -0.163 | +0.353 |
| +/-5 pp | -0.786 | -0.221 | +0.376 |

### `utility` (consistent sign, variable magnitude)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.901 | +0.808 | +0.342 |
| +/-2 pp | +0.886 | +0.729 | -0.088 |
| +/-3 pp | +0.879 | +0.726 | -0.246 |
| +/-5 pp | +0.877 | +0.761 | -0.228 |

### `capacity` (consistent sign, weak in at least one cell)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | -0.429 | -0.080 | n<8 |
| +/-2 pp | -0.590 | -0.148 | n<8 |
| +/-3 pp | -0.573 | -0.310 | n<8 |
| +/-5 pp | -0.566 | -0.478 | n<8 |

### `center alignment` (robust)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.483 | +0.581 | n<8 |
| +/-2 pp | +0.802 | +0.623 | n<8 |
| +/-3 pp | +0.815 | +0.633 | n<8 |
| +/-5 pp | +0.808 | +0.701 | n<8 |

### `participation ratio (per class)` (robust)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.659 | +0.592 | +0.947 |
| +/-2 pp | +0.764 | +0.659 | +0.942 |
| +/-3 pp | +0.770 | +0.720 | +0.947 |
| +/-5 pp | +0.763 | +0.779 | +0.951 |

### `participation ratio (global)` (robust)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.877 | +0.734 | +0.895 |
| +/-2 pp | +0.894 | +0.763 | +0.860 |
| +/-3 pp | +0.896 | +0.806 | +0.872 |
| +/-5 pp | +0.891 | +0.842 | +0.879 |

### `numerical rank (per class)` (consistent sign, variable magnitude)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.436 | n<8 | +0.972 |
| +/-2 pp | +0.607 | n<8 | +0.958 |
| +/-3 pp | +0.597 | n<8 | +0.950 |
| +/-5 pp | +0.597 | n<8 | +0.944 |

### `numerical rank (global)` (insufficient data)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.556 | n<8 | n<8 |
| +/-2 pp | +0.703 | n<8 | n<8 |
| +/-3 pp | +0.687 | n<8 | n<8 |
| +/-5 pp | +0.686 | n<8 | n<8 |

### `sparsity` (consistent sign, weak in at least one cell)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.268 | +0.390 | n<8 |
| +/-2 pp | +0.619 | +0.450 | n<8 |
| +/-3 pp | +0.604 | +0.395 | n<8 |
| +/-5 pp | +0.604 | +0.376 | n<8 |

### `mean distance` (consistent sign, weak in at least one cell)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.124 | +0.485 | n<8 |
| +/-2 pp | +0.154 | +0.546 | n<8 |
| +/-3 pp | +0.146 | +0.611 | n<8 |
| +/-5 pp | +0.145 | +0.696 | n<8 |

### `mean angle` (SIGN FLIPS (Pearson))

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.124 | -0.102 | n<8 |
| +/-2 pp | +0.351 | -0.180 | n<8 |
| +/-3 pp | +0.361 | -0.165 | n<8 |
| +/-5 pp | +0.363 | -0.339 | n<8 |

### `mean covariance` (consistent sign, weak in at least one cell)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | +0.035 | +0.405 | n<8 |
| +/-2 pp | +0.013 | +0.489 | n<8 |
| +/-3 pp | +0.017 | +0.516 | n<8 |
| +/-5 pp | +0.018 | +0.556 | n<8 |

### `NC1` (SIGN FLIPS (Spearman))

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | -0.145 | n<8 | -0.178 |
| +/-2 pp | -0.044 | n<8 | -0.149 |
| +/-3 pp | -0.040 | n<8 | -0.308 |
| +/-5 pp | -0.036 | n<8 | -0.296 |

### `local smoothness` (insufficient data)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | n<8 | n<8 | +0.931 |
| +/-2 pp | n<8 | n<8 | +0.889 |
| +/-3 pp | n<8 | n<8 | +0.891 |
| +/-5 pp | n<8 | n<8 | +0.899 |

### `effective dim (PCA)` (insufficient data)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | n<8 | n<8 | +0.856 |
| +/-2 pp | n<8 | n<8 | +0.790 |
| +/-3 pp | n<8 | n<8 | +0.788 |
| +/-5 pp | n<8 | n<8 | +0.805 |

### `class separability` (insufficient data)

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| +/-1 pp | n<8 | n<8 | -0.832 |
| +/-2 pp | n<8 | n<8 | -0.767 |
| +/-3 pp | n<8 | n<8 | -0.788 |
| +/-5 pp | n<8 | n<8 | -0.805 |

## Summary

- robust: 4
- consistent sign, weak in at least one cell: 4
- insufficient data: 4
- SIGN FLIPS (Pearson): 2
- consistent sign, variable magnitude: 2
- SIGN FLIPS (Spearman): 1
