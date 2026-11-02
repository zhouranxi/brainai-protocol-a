# Cross-implementation stability of the GLUE markers

Protocol: ID-matched around the mean of `acc_test` (band in percentage points),
then architecture-residualised. Three cells: the official implementation on the
official 224-model release, the official implementation on the official 80-model
release, and our own implementation on our own 48 models.

## The matrix (band = 1 pp)

| Marker | official 224 | official 80 | our own 48 | sign consistent |
|---|---|---|---|---|
| `dimension` | r=+0.925 rho=+0.847 (n=106) | r=+0.737 rho=+0.803 (n=25) | r=+0.919 rho=+0.868 (n=32) | yes |
| `radius` | r=-0.584 rho=-0.639 (n=106) | r=+0.153 rho=+0.322 (n=25) | r=+0.602 rho=+0.673 (n=32) | NO |
| `utility` | r=+0.901 rho=+0.769 (n=106) | r=+0.808 rho=+0.896 (n=25) | r=+0.342 rho=+0.135 (n=32) | yes |

## Band sensitivity

### `dimension`

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| ±1 pp | r=+0.925 (n=106) | r=+0.737 (n=25) | r=+0.919 (n=32) |
| ±2 pp | r=+0.911 (n=194) | r=+0.713 (n=43) | r=+0.903 (n=46) |
| ±3 pp | r=+0.906 (n=214) | r=+0.751 (n=65) | r=+0.901 (n=47) |
| ±5 pp | r=+0.904 (n=223) | r=+0.776 (n=80) | r=+0.908 (n=48) |

### `radius`

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| ±1 pp | r=-0.584 (n=106) | r=+0.153 (n=25) | r=+0.602 (n=32) |
| ±2 pp | r=-0.796 (n=194) | r=+0.025 (n=43) | r=+0.492 (n=46) |
| ±3 pp | r=-0.790 (n=214) | r=-0.163 (n=65) | r=+0.353 (n=47) |
| ±5 pp | r=-0.786 (n=223) | r=-0.221 (n=80) | r=+0.376 (n=48) |

### `utility`

| band | official 224 | official 80 | our own 48 |
|---|---|---|---|
| ±1 pp | r=+0.901 (n=106) | r=+0.808 (n=25) | r=+0.342 (n=32) |
| ±2 pp | r=+0.886 (n=194) | r=+0.729 (n=43) | r=-0.088 (n=46) |
| ±3 pp | r=+0.879 (n=214) | r=+0.726 (n=65) | r=-0.246 (n=47) |
| ±5 pp | r=+0.877 (n=223) | r=+0.761 (n=80) | r=-0.228 (n=48) |
