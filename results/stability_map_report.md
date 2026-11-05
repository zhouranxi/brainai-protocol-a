# Sign-stability map: geometry markers vs OOD, full design

Protocol: ID-match (±1 pp around mean `acc_test`) + architecture
residualisation (own48 is single-architecture → mean-centering).
Usable = n≥8, |r|≥0.05. Groups pair implementations:
`task` = official224 + own48/c100 + own48/tinyimg;
`corruption:<name>` = official80/<name> + own48/<name>.
resample-agree = fraction of seeded half-samples keeping the
full-sample sign (1 pp band, gaussian_noise / task cells).

## Verdict summary

| verdict | # families |
|---|---|
| stable-same | 1 |
| weak-same | 1 |
| flip | 2 |
| shift-dependent | 18 |
| unstable-n | 2 |
| na | 0 |

## Per-family matrix (primary band = 1 pp)

### `axis_alignment` — **stable-same** (official) [band_flip_groups=] [single_source] [magnitude_collapse]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:axis_alignment | -0.065 | -0.009 | 25 | — |
| corruption:contrast | official80/contrast:axis_alignment | -0.169 | -0.175 | 25 | — |
| corruption:fog | official80/fog:axis_alignment | -0.050 | -0.018 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:axis_alignment | -0.166 | -0.119 | 25 | 0.71 |
| corruption:jpeg_compression | official80/jpeg_compression:axis_alignment | -0.309 | -0.317 | 25 | — |
| corruption:pixelate | official80/pixelate:axis_alignment | -0.345 | -0.318 | 25 | — |
| task | official224:axis_alignment | -0.412 | -0.373 | 106 | 1.00 |

### `capacity` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:capacity | +0.366 | +0.353 | 25 | — |
| corruption:contrast | official80/contrast:capacity | +0.512 | +0.442 | 25 | — |
| corruption:fog | official80/fog:capacity | +0.509 | +0.443 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:capacity | -0.080 | -0.200 | 25 | 0.56 |
| corruption:jpeg_compression | official80/jpeg_compression:capacity | -0.081 | -0.106 | 25 | — |
| corruption:pixelate | official80/pixelate:capacity | -0.236 | -0.439 | 25 | — |
| task | official224:capacity | -0.429 | -0.331 | 106 | 1.00 |

### `center_alignment` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:center_alignment | -0.650 | -0.566 | 25 | — |
| corruption:contrast | official80/contrast:center_alignment | -0.604 | -0.498 | 25 | — |
| corruption:fog | official80/fog:center_alignment | -0.646 | -0.561 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:center_alignment | +0.581 | +0.642 | 25 | 1.00 |
| corruption:jpeg_compression | official80/jpeg_compression:center_alignment | +0.089 | -0.040 | 25 | — |
| corruption:pixelate | official80/pixelate:center_alignment | +0.179 | +0.132 | 25 | — |
| task | official224:center_alignment | +0.483 | +0.400 | 106 | 1.00 |

### `center_axis_alignment` — **unstable-n** (official) [band_flip_groups=corruption:gaussian_noise] [single_source] [magnitude_collapse]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:center_axis_alignment | -0.149 | -0.100 | 25 | — |
| corruption:contrast | official80/contrast:center_axis_alignment | -0.247 | -0.196 | 25 | — |
| corruption:fog | official80/fog:center_axis_alignment | -0.258 | -0.195 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:center_axis_alignment | -0.285 | -0.196 | 25 | 0.80 |
| corruption:jpeg_compression | official80/jpeg_compression:center_axis_alignment | -0.393 | -0.410 | 25 | — |
| corruption:pixelate | official80/pixelate:center_axis_alignment | -0.237 | -0.110 | 25 | — |
| task | official224:center_axis_alignment | -0.154 | -0.358 | 106 | 1.00 |

### `class_separability` — **shift-dependent** (own) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=own]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | own48/brightness:id_class_separability | +0.584 | +0.590 | 32 | 1.00 |
| corruption:contrast | own48/contrast:id_class_separability | +0.168 | +0.226 | 32 | 0.84 |
| corruption:fog | own48/fog:id_class_separability | +0.472 | +0.508 | 32 | 1.00 |
| corruption:gaussian_noise | own48/gaussian_noise:id_class_separability | -0.552 | -0.537 | 32 | 1.00 |
| corruption:jpeg_compression | own48/jpeg_compression:id_class_separability | -0.165 | -0.143 | 32 | 0.76 |
| corruption:pixelate | own48/pixelate:id_class_separability | -0.240 | -0.266 | 32 | 0.93 |
| task | own48/c100:id_class_separability | -0.832 | -0.799 | 32 | 1.00 |
| task | own48/tinyimg:id_class_separability | -0.843 | -0.834 | 32 | 1.00 |

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:id_class_separability | -0.933 | 12 |
| own48/lr_tier:lr0.25:id_class_separability | -0.861 | 12 |
| own48/lr_tier:lr0.5:id_class_separability | -0.469 | 12 |
| own48/lr_tier:lr1.0:id_class_separability | -0.824 | 12 |

### `dimension` — **shift-dependent** (official+own) [band_flip_groups=] [shift_flips=official/own]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:dimension | -0.493 | -0.607 | 25 | — |
| corruption:brightness | own48/brightness:glue_dimension | -0.712 | -0.561 | 32 | 1.00 |
| corruption:contrast | official80/contrast:dimension | -0.584 | -0.626 | 25 | — |
| corruption:contrast | own48/contrast:glue_dimension | -0.114 | -0.172 | 32 | 0.73 |
| corruption:fog | official80/fog:dimension | -0.562 | -0.649 | 25 | — |
| corruption:fog | own48/fog:glue_dimension | -0.620 | -0.683 | 32 | 1.00 |
| corruption:gaussian_noise | official80/gaussian_noise:dimension | +0.737 | +0.803 | 25 | 1.00 |
| corruption:gaussian_noise | own48/gaussian_noise:glue_dimension | +0.623 | +0.633 | 32 | 1.00 |
| corruption:jpeg_compression | official80/jpeg_compression:dimension | +0.592 | +0.289 | 25 | — |
| corruption:jpeg_compression | own48/jpeg_compression:glue_dimension | — | +0.092 | 32 | — |
| corruption:pixelate | official80/pixelate:dimension | +0.516 | +0.128 | 25 | — |
| corruption:pixelate | own48/pixelate:glue_dimension | +0.112 | +0.066 | 32 | 0.75 |
| task | official224:dimension | +0.925 | +0.847 | 106 | 1.00 |
| task | own48/c100:glue_dimension | +0.919 | +0.868 | 32 | 1.00 |
| task | own48/tinyimg:glue_dimension | +0.929 | +0.864 | 32 | 1.00 |

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:glue_dimension | +0.911 | 12 |
| own48/lr_tier:lr0.25:glue_dimension | +0.796 | 12 |
| own48/lr_tier:lr0.5:glue_dimension | +0.612 | 12 |
| own48/lr_tier:lr1.0:glue_dimension | +0.697 | 12 |

### `effective_dim_pca` — **shift-dependent** (own) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=own]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | own48/brightness:id_effective_dim_pca | -0.693 | -0.536 | 32 | 0.99 |
| corruption:contrast | own48/contrast:id_effective_dim_pca | -0.129 | -0.173 | 32 | 0.79 |
| corruption:fog | own48/fog:id_effective_dim_pca | -0.580 | -0.582 | 32 | 1.00 |
| corruption:gaussian_noise | own48/gaussian_noise:id_effective_dim_pca | +0.559 | +0.671 | 32 | 1.00 |
| corruption:jpeg_compression | own48/jpeg_compression:id_effective_dim_pca | — | +0.168 | 32 | — |
| corruption:pixelate | own48/pixelate:id_effective_dim_pca | +0.087 | +0.377 | 32 | 0.72 |
| task | own48/c100:id_effective_dim_pca | +0.856 | +0.936 | 32 | 1.00 |
| task | own48/tinyimg:id_effective_dim_pca | +0.870 | +0.941 | 32 | 1.00 |

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:id_effective_dim_pca | +0.889 | 12 |
| own48/lr_tier:lr0.25:id_effective_dim_pca | +0.940 | 12 |
| own48/lr_tier:lr0.5:id_effective_dim_pca | +0.450 | 12 |
| own48/lr_tier:lr1.0:id_effective_dim_pca | +0.916 | 12 |

### `local_smoothness` — **shift-dependent** (own) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=own]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | own48/brightness:id_local_smoothness | -0.663 | -0.554 | 32 | 0.99 |
| corruption:contrast | own48/contrast:id_local_smoothness | -0.168 | -0.153 | 32 | 0.83 |
| corruption:fog | own48/fog:id_local_smoothness | -0.592 | -0.590 | 32 | 1.00 |
| corruption:gaussian_noise | own48/gaussian_noise:id_local_smoothness | +0.646 | +0.717 | 32 | 1.00 |
| corruption:jpeg_compression | own48/jpeg_compression:id_local_smoothness | +0.151 | +0.210 | 32 | 0.73 |
| corruption:pixelate | own48/pixelate:id_local_smoothness | +0.201 | +0.328 | 32 | 0.90 |
| task | own48/c100:id_local_smoothness | +0.931 | +0.949 | 32 | 1.00 |
| task | own48/tinyimg:id_local_smoothness | +0.937 | +0.957 | 32 | 1.00 |

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:id_local_smoothness | +0.923 | 12 |
| own48/lr_tier:lr0.25:id_local_smoothness | +0.950 | 12 |
| own48/lr_tier:lr0.5:id_local_smoothness | +0.741 | 12 |
| own48/lr_tier:lr1.0:id_local_smoothness | +0.983 | 12 |

### `mean_angle` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:mean_angle | +0.393 | +0.294 | 25 | — |
| corruption:contrast | official80/contrast:mean_angle | +0.614 | +0.597 | 25 | — |
| corruption:fog | official80/fog:mean_angle | +0.558 | +0.400 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:mean_angle | -0.102 | -0.152 | 25 | 0.72 |
| corruption:jpeg_compression | official80/jpeg_compression:mean_angle | -0.128 | -0.219 | 25 | — |
| corruption:pixelate | official80/pixelate:mean_angle | -0.092 | -0.190 | 25 | — |
| task | official224:mean_angle | +0.124 | -0.121 | 106 | 0.81 |

### `mean_angle_avg` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:mean_angle_avg | -0.692 | -0.655 | 25 | — |
| corruption:contrast | official80/contrast:mean_angle_avg | -0.694 | -0.613 | 25 | — |
| corruption:fog | official80/fog:mean_angle_avg | -0.770 | -0.711 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:mean_angle_avg | +0.669 | +0.755 | 25 | 1.00 |
| corruption:jpeg_compression | official80/jpeg_compression:mean_angle_avg | +0.183 | +0.084 | 25 | — |
| corruption:pixelate | official80/pixelate:mean_angle_avg | +0.074 | -0.052 | 25 | — |
| task | official224:mean_angle_avg | +0.826 | +0.650 | 106 | 1.00 |

### `mean_covariance` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:mean_covariance | -0.453 | -0.421 | 25 | — |
| corruption:contrast | official80/contrast:mean_covariance | -0.662 | -0.622 | 25 | — |
| corruption:fog | official80/fog:mean_covariance | -0.684 | -0.598 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:mean_covariance | +0.405 | +0.561 | 25 | 0.95 |
| corruption:jpeg_compression | official80/jpeg_compression:mean_covariance | +0.123 | +0.227 | 25 | — |
| corruption:pixelate | official80/pixelate:mean_covariance | +0.090 | +0.105 | 25 | — |
| task | official224:mean_covariance | — | +0.444 | 106 | — |

### `mean_covariance_avg` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:mean_covariance_avg | -0.110 | -0.316 | 25 | — |
| corruption:contrast | official80/contrast:mean_covariance_avg | -0.410 | -0.606 | 25 | — |
| corruption:fog | official80/fog:mean_covariance_avg | -0.367 | -0.501 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:mean_covariance_avg | +0.182 | +0.467 | 25 | 0.73 |
| corruption:jpeg_compression | official80/jpeg_compression:mean_covariance_avg | +0.217 | +0.284 | 25 | — |
| corruption:pixelate | official80/pixelate:mean_covariance_avg | +0.193 | +0.092 | 25 | — |
| task | official224:mean_covariance_avg | — | +0.445 | 106 | — |

### `mean_distance` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:mean_distance | -0.608 | -0.606 | 25 | — |
| corruption:contrast | official80/contrast:mean_distance | -0.716 | -0.628 | 25 | — |
| corruption:fog | official80/fog:mean_distance | -0.783 | -0.739 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:mean_distance | +0.485 | +0.607 | 25 | 0.98 |
| corruption:jpeg_compression | official80/jpeg_compression:mean_distance | — | -0.035 | 25 | — |
| corruption:pixelate | official80/pixelate:mean_distance | — | -0.097 | 25 | — |
| task | official224:mean_distance | +0.124 | +0.296 | 106 | 1.00 |

### `mean_distance_avg` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:mean_distance_avg | -0.642 | -0.642 | 25 | — |
| corruption:contrast | official80/contrast:mean_distance_avg | -0.749 | -0.694 | 25 | — |
| corruption:fog | official80/fog:mean_distance_avg | -0.815 | -0.791 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:mean_distance_avg | +0.522 | +0.668 | 25 | 0.99 |
| corruption:jpeg_compression | official80/jpeg_compression:mean_distance_avg | — | -0.007 | 25 | — |
| corruption:pixelate | official80/pixelate:mean_distance_avg | — | -0.108 | 25 | — |
| task | official224:mean_distance_avg | +0.111 | +0.389 | 106 | 1.00 |

### `nc1` — **unstable-n** (official+own) [band_flip_groups=corruption:contrast/corruption:fog/corruption:jpeg_compression/corruption:pixelate] [magnitude_collapse] [shift_flips=own]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | own48/brightness:id_nc1 | — | -0.309 | 32 | — |
| corruption:contrast | own48/contrast:id_nc1 | -0.064 | -0.313 | 32 | 0.93 |
| corruption:fog | own48/fog:id_nc1 | +0.114 | -0.210 | 32 | 0.51 |
| corruption:gaussian_noise | own48/gaussian_noise:id_nc1 | -0.344 | -0.005 | 32 | 0.71 |
| corruption:jpeg_compression | own48/jpeg_compression:id_nc1 | +0.184 | -0.063 | 32 | 0.65 |
| corruption:pixelate | own48/pixelate:id_nc1 | +0.148 | +0.087 | 32 | 0.73 |
| task | official224:nc1 | -0.145 | -0.146 | 106 | 0.97 |
| task | own48/c100:id_nc1 | -0.178 | +0.097 | 32 | 0.61 |
| task | own48/tinyimg:id_nc1 | -0.159 | +0.079 | 32 | 0.60 |

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:id_nc1 | +0.861 | 12 |
| own48/lr_tier:lr0.25:id_nc1 | +0.852 | 12 |
| own48/lr_tier:lr0.5:id_nc1 | +0.528 | 12 |
| own48/lr_tier:lr1.0:id_nc1 | -0.733 | 12 |

### `numerical_rank` — **shift-dependent** (official+own) [band_flip_groups=] [magnitude_collapse] [shift_flips=own]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | own48/brightness:id_numerical_rank | -0.703 | -0.589 | 32 | 1.00 |
| corruption:contrast | own48/contrast:id_numerical_rank | -0.092 | -0.180 | 32 | 0.71 |
| corruption:fog | own48/fog:id_numerical_rank | -0.578 | -0.581 | 32 | 1.00 |
| corruption:gaussian_noise | own48/gaussian_noise:id_numerical_rank | +0.708 | +0.734 | 32 | 1.00 |
| corruption:jpeg_compression | own48/jpeg_compression:id_numerical_rank | +0.117 | +0.143 | 32 | 0.68 |
| corruption:pixelate | own48/pixelate:id_numerical_rank | +0.262 | +0.313 | 32 | 0.95 |
| task | official224:numerical_rank | +0.436 | +0.570 | 106 | 1.00 |
| task | own48/c100:id_numerical_rank | +0.972 | +0.970 | 32 | 1.00 |
| task | own48/tinyimg:id_numerical_rank | +0.966 | +0.975 | 32 | 1.00 |

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:id_numerical_rank | +0.966 | 12 |
| own48/lr_tier:lr0.25:id_numerical_rank | +0.904 | 12 |
| own48/lr_tier:lr0.5:id_numerical_rank | +0.832 | 12 |
| own48/lr_tier:lr1.0:id_numerical_rank | +0.802 | 12 |

### `numerical_rank_avg` — **weak-same** (official) [band_flip_groups=] [single_source]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| task | official224:numerical_rank_avg | +0.556 | +0.633 | 106 | 1.00 |

### `participation_ratio` — **shift-dependent** (official+own) [band_flip_groups=] [magnitude_collapse] [shift_flips=official/own]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:participation_ratio | -0.532 | -0.593 | 25 | — |
| corruption:brightness | own48/brightness:id_pr_global | -0.622 | -0.545 | 32 | 1.00 |
| corruption:contrast | official80/contrast:participation_ratio | -0.308 | -0.495 | 25 | — |
| corruption:contrast | own48/contrast:id_pr_global | -0.154 | -0.152 | 32 | 0.84 |
| corruption:fog | official80/fog:participation_ratio | -0.377 | -0.562 | 25 | — |
| corruption:fog | own48/fog:id_pr_global | -0.526 | -0.502 | 32 | 0.99 |
| corruption:gaussian_noise | official80/gaussian_noise:participation_ratio | +0.592 | +0.735 | 25 | 1.00 |
| corruption:gaussian_noise | own48/gaussian_noise:id_pr_global | +0.630 | +0.686 | 32 | 1.00 |
| corruption:jpeg_compression | official80/jpeg_compression:participation_ratio | +0.281 | +0.139 | 25 | — |
| corruption:jpeg_compression | own48/jpeg_compression:id_pr_global | +0.157 | +0.199 | 32 | 0.79 |
| corruption:pixelate | official80/pixelate:participation_ratio | +0.277 | +0.233 | 25 | — |
| corruption:pixelate | own48/pixelate:id_pr_global | +0.218 | +0.393 | 32 | 0.93 |
| task | official224:participation_ratio | +0.659 | +0.503 | 106 | 1.00 |
| task | own48/c100:id_pr_global | +0.895 | +0.901 | 32 | 1.00 |
| task | own48/tinyimg:id_pr_global | +0.901 | +0.907 | 32 | 1.00 |

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:id_pr_global | +0.899 | 12 |
| own48/lr_tier:lr0.25:id_pr_global | +0.925 | 12 |
| own48/lr_tier:lr0.5:id_pr_global | +0.301 | 12 |
| own48/lr_tier:lr1.0:id_pr_global | +0.922 | 12 |

### `participation_ratio_avg` — **shift-dependent** (official) [band_flip_groups=] [single_source] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:participation_ratio_avg | -0.541 | -0.609 | 25 | — |
| corruption:contrast | official80/contrast:participation_ratio_avg | -0.580 | -0.593 | 25 | — |
| corruption:fog | official80/fog:participation_ratio_avg | -0.545 | -0.565 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:participation_ratio_avg | +0.734 | +0.878 | 25 | 1.00 |
| corruption:jpeg_compression | official80/jpeg_compression:participation_ratio_avg | +0.583 | +0.372 | 25 | — |
| corruption:pixelate | official80/pixelate:participation_ratio_avg | +0.554 | +0.272 | 25 | — |
| task | official224:participation_ratio_avg | +0.877 | +0.700 | 106 | 1.00 |

### `participation_ratio_perclass` — **shift-dependent** (own) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=own]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | own48/brightness:id_pr_perclass | -0.658 | -0.455 | 32 | 0.99 |
| corruption:contrast | own48/contrast:id_pr_perclass | -0.085 | -0.082 | 32 | 0.70 |
| corruption:fog | own48/fog:id_pr_perclass | -0.559 | -0.484 | 32 | 1.00 |
| corruption:gaussian_noise | own48/gaussian_noise:id_pr_perclass | +0.653 | +0.700 | 32 | 1.00 |
| corruption:jpeg_compression | own48/jpeg_compression:id_pr_perclass | +0.127 | +0.226 | 32 | 0.68 |
| corruption:pixelate | own48/pixelate:id_pr_perclass | +0.226 | +0.231 | 32 | 0.93 |
| task | own48/c100:id_pr_perclass | +0.947 | +0.922 | 32 | 1.00 |
| task | own48/tinyimg:id_pr_perclass | +0.944 | +0.872 | 32 | 1.00 |

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:id_pr_perclass | +0.961 | 12 |
| own48/lr_tier:lr0.25:id_pr_perclass | +0.911 | 12 |
| own48/lr_tier:lr0.5:id_pr_perclass | -0.345 | 12 |
| own48/lr_tier:lr1.0:id_pr_perclass | +0.921 | 12 |

### `radius` — **flip** (official+own)

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:radius | -0.385 | -0.493 | 25 | — |
| corruption:brightness | own48/brightness:glue_radius | -0.412 | -0.437 | 32 | 1.00 |
| corruption:contrast | official80/contrast:radius | -0.262 | -0.487 | 25 | — |
| corruption:contrast | own48/contrast:glue_radius | -0.187 | -0.178 | 32 | 0.80 |
| corruption:fog | official80/fog:radius | -0.272 | -0.505 | 25 | — |
| corruption:fog | own48/fog:glue_radius | -0.340 | -0.416 | 32 | 0.98 |
| corruption:gaussian_noise | official80/gaussian_noise:radius | +0.153 | +0.322 | 25 | 0.77 |
| corruption:gaussian_noise | own48/gaussian_noise:glue_radius | +0.438 | +0.435 | 32 | 1.00 |
| corruption:jpeg_compression | official80/jpeg_compression:radius | -0.185 | -0.163 | 25 | — |
| corruption:jpeg_compression | own48/jpeg_compression:glue_radius | +0.204 | +0.124 | 32 | 0.88 |
| corruption:pixelate | official80/pixelate:radius | -0.283 | -0.272 | 25 | — |
| corruption:pixelate | own48/pixelate:glue_radius | +0.301 | +0.243 | 32 | 0.97 |
| task | official224:radius | -0.584 | -0.639 | 106 | 1.00 |
| task | own48/c100:glue_radius | +0.602 | +0.673 | 32 | 1.00 |
| task | own48/tinyimg:glue_radius | +0.626 | +0.735 | 32 | 1.00 |
- **flip in `corruption:jpeg_compression`**: {'official80/jpeg_compression:radius': -0.185, 'own48/jpeg_compression:glue_radius': 0.204}
- **flip in `corruption:pixelate`**: {'official80/pixelate:radius': -0.283, 'own48/pixelate:glue_radius': 0.301}
- **flip in `task`**: {'official224:radius': -0.584, 'own48/c100:glue_radius': 0.602, 'own48/tinyimg:glue_radius': 0.626}

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:glue_radius | +0.796 | 12 |
| own48/lr_tier:lr0.25:glue_radius | +0.558 | 12 |
| own48/lr_tier:lr0.5:glue_radius | +0.779 | 12 |
| own48/lr_tier:lr1.0:glue_radius | -0.422 | 12 |

### `sparsity` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:sparsity | -0.197 | -0.369 | 25 | — |
| corruption:contrast | official80/contrast:sparsity | -0.411 | -0.558 | 25 | — |
| corruption:fog | official80/fog:sparsity | -0.305 | -0.503 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:sparsity | +0.390 | +0.432 | 25 | 0.90 |
| corruption:jpeg_compression | official80/jpeg_compression:sparsity | +0.685 | +0.263 | 25 | — |
| corruption:pixelate | official80/pixelate:sparsity | +0.628 | +0.185 | 25 | — |
| task | official224:sparsity | +0.268 | +0.584 | 106 | 1.00 |

### `sparsity_avg` — **shift-dependent** (official) [band_flip_groups=] [single_source] [magnitude_collapse] [shift_flips=official]

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:sparsity_avg | -0.197 | -0.369 | 25 | — |
| corruption:contrast | official80/contrast:sparsity_avg | -0.411 | -0.558 | 25 | — |
| corruption:fog | official80/fog:sparsity_avg | -0.305 | -0.503 | 25 | — |
| corruption:gaussian_noise | official80/gaussian_noise:sparsity_avg | +0.390 | +0.432 | 25 | 0.90 |
| corruption:jpeg_compression | official80/jpeg_compression:sparsity_avg | +0.685 | +0.263 | 25 | — |
| corruption:pixelate | official80/pixelate:sparsity_avg | +0.628 | +0.185 | 25 | — |
| task | official224:sparsity_avg | +0.268 | +0.584 | 106 | 1.00 |

### `utility` — **flip** (official+own)

| group | cell | r | rho | n | resample-agree |
|---|---|---|---|---|---|
| corruption:brightness | official80/brightness:utility | -0.562 | -0.626 | 25 | — |
| corruption:brightness | own48/brightness:glue_utility | -0.419 | -0.363 | 32 | 0.99 |
| corruption:contrast | official80/contrast:utility | -0.608 | -0.688 | 25 | — |
| corruption:contrast | own48/contrast:glue_utility | — | -0.160 | 32 | — |
| corruption:fog | official80/fog:utility | -0.596 | -0.655 | 25 | — |
| corruption:fog | own48/fog:glue_utility | -0.178 | -0.191 | 32 | 0.73 |
| corruption:gaussian_noise | official80/gaussian_noise:utility | +0.808 | +0.896 | 25 | 1.00 |
| corruption:gaussian_noise | own48/gaussian_noise:glue_utility | +0.061 | +0.040 | 32 | 0.62 |
| corruption:jpeg_compression | official80/jpeg_compression:utility | +0.542 | +0.366 | 25 | — |
| corruption:jpeg_compression | own48/jpeg_compression:glue_utility | -0.133 | -0.077 | 32 | 0.71 |
| corruption:pixelate | official80/pixelate:utility | +0.411 | +0.114 | 25 | — |
| corruption:pixelate | own48/pixelate:glue_utility | -0.178 | -0.190 | 32 | 0.89 |
| task | official224:utility | +0.901 | +0.769 | 106 | 1.00 |
| task | own48/c100:glue_utility | +0.342 | +0.135 | 32 | 0.92 |
| task | own48/tinyimg:glue_utility | +0.384 | +0.240 | 32 | 0.94 |
- **flip in `corruption:jpeg_compression`**: {'official80/jpeg_compression:utility': 0.542, 'own48/jpeg_compression:glue_utility': -0.133}
- **flip in `corruption:pixelate`**: {'official80/pixelate:utility': 0.411, 'own48/pixelate:glue_utility': -0.178}

| aux (severity, not verdict-bearing) | r | n |
|---|---|---|
| own48/lr_tier:lr0.125:glue_utility | +0.881 | 12 |
| own48/lr_tier:lr0.25:glue_utility | +0.746 | 12 |
| own48/lr_tier:lr0.5:glue_utility | +0.165 | 12 |
| own48/lr_tier:lr1.0:glue_utility | -0.854 | 12 |
