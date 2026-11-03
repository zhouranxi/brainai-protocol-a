# 论文数字一致性审计（paper_draft_v1.md vs 原始数据重算）

生成：2026-10-27 · 共 411 条断言 · 通过 411 · 不一致 0

方法：对每条论文声称的数字，直接从发布数据（df_fig3.pkl / official_df_fig15-20.pkl /
ind_validation_results.json / mod_arch_results.json / glue_own48_results.json）用与
分析脚本相同的协议（OLS dummy 残差化、Spearman、分组符号一致性）独立重算，逐条比对。

## 全部断言

| 节 | 量 | 论文值 | 重算值 | 判定 |
|---|---|---|---|---|
| 4.2 | D_eff vs OOD (full sample) | 0.94 | 0.9405 | ✓ |
| 4.2 | D_eff vs ID acc (full sample) | 0.5 | 0.5019 | ✓ |
| 4.2 | PR vs OOD (full sample) | 0.88 | 0.8748 | ✓ |
| 4.2 | PR vs ID acc (full sample) | 0.42 | 0.4191 | ✓ |
| Table2 | D_eff subset r (pre-C2) | 0.932 | 0.9321 | ✓ |
| Table2 | D_eff resid r (C1+C2) | 0.925 | 0.925 | ✓ |
| Table2 | D_eff Spearman rho | 0.847 | 0.8474 | ✓ |
| Table2 | Psi_eff subset r (pre-C2) | 0.934 | 0.9344 | ✓ |
| Table2 | Psi_eff resid r (C1+C2) | 0.901 | 0.9014 | ✓ |
| Table2 | Psi_eff Spearman rho | 0.769 | 0.7688 | ✓ |
| Table2 | R_eff subset r (pre-C2) | -0.547 | -0.547 | ✓ |
| Table2 | R_eff resid r (C1+C2) | -0.584 | -0.5843 | ✓ |
| Table2 | R_eff Spearman rho | -0.639 | -0.6393 | ✓ |
| Table2 | PR subset r (pre-C2) | 0.858 | 0.8578 | ✓ |
| Table2 | PR resid r (C1+C2) | 0.659 | 0.6586 | ✓ |
| Table2 | PR Spearman rho | 0.503 | 0.5026 | ✓ |
| Table2 | numrank subset r (pre-C2) | -0.212 | -0.2119 | ✓ |
| Table2 | numrank resid r (C1+C2) | 0.436 | 0.4363 | ✓ |
| Table2 | numrank Spearman rho | 0.57 | 0.5697 | ✓ |
| Table2 | NC1 subset r (pre-C2) | 0.019 | 0.0193 | ✓ |
| Table2 | NC1 resid r (C1+C2) | -0.145 | -0.145 | ✓ |
| Table2 | NC1 Spearman rho | -0.146 | -0.1459 | ✓ |
| Table2 | ID acc subset r (pre-C2) | -0.045 | -0.0447 | ✓ |
| Table2 | ID acc resid r (C1+C2) | 0.392 | 0.3922 | ✓ |
| Table2 | ID acc Spearman rho | 0.365 | 0.3652 | ✓ |
| Table2 | n_subset | 106 | 106 | ✓ |
| Table2 | ID span (pp) | 1.92 | 1.92 | ✓ |
| Table2 | OOD span preserved % | 91.8 | 91.7564 | ✓ |
| Table3 | ID acc per-arch signs | +0.48/-0.48/+0.46/+0.24/+0.42 | +0.48/-0.48/+0.46/+0.24/+0.42 | ✓ |
| 3.3 | no-C1: 80-model pooled r | 0.44 | 0.4426 | ✓ |
| 3.3 | no-C1: 80-model C1+C2 r | -0.63 | -0.6281 | ✓ |
| 3.3 | no-C3: center_alignment r | 0.48 | 0.4829 | ✓ |
| 3.3 | no-C3: center_alignment rho | 0.4 | 0.4003 | ✓ |
| 3.3 | no-C4: PR resid r | 0.66 | 0.6586 | ✓ |
| 4.4 | D_eff band range (224) | 0.89-0.93 | 0.9035 | ✓ |
| 4.4 | D_eff band range (80) | 0.71-0.78 | 0.7128 | ✓ |
| 4.4 | baseline band range (224) | 0.39-0.57 | 0.3922 | ✓ |
| 4.4 | baseline band range (80) | -0.63--0.77 | -0.7671 | ✓ |
| 4.7 | numrank C100 | 0.87 | 0.8698 | ✓ |
| 4.7 | numrank Tiny | 0.853 | 0.8526 | ✓ |
| 4.7 | smoothness C100 | 0.832 | 0.8323 | ✓ |
| 4.7 | smoothness Tiny | 0.792 | 0.7925 | ✓ |
| 4.7 | PR C100 | 0.701 | 0.7009 | ✓ |
| 4.7 | PR Tiny | 0.707 | 0.7068 | ✓ |
| 4.7 | ID acc C100 | -0.439 | -0.4392 | ✓ |
| 4.7 | ID acc Tiny | -0.338 | -0.3377 | ✓ |
| 4.7-C4 | baseline per-family lr=0.125 | -0.802 | -0.8015 | ✓ |
| 4.7-C4 | baseline per-family lr=0.25 | -0.603 | -0.6032 | ✓ |
| 4.7-C4 | baseline per-family lr=0.5 | -0.33 | -0.3303 | ✓ |
| 4.7-C4 | baseline per-family lr=1.0 | 0.694 | 0.6944 | ✓ |
| 4.7-conv | converged per-family lr=0.125 | -0.966 | -0.9658 | ✓ |
| 4.7-conv | converged per-family lr=0.25 | -0.603 | -0.6032 | ✓ |
| 4.7-conv | converged per-family lr=0.5 | -0.33 | -0.3303 | ✓ |
| 4.7-conv | converged baseline aggregate C100 | -0.658 | -0.6577 | ✓ |
| 4.7-conv | converged baseline aggregate Tiny | -0.513 | -0.5132 | ✓ |
| 4.7c | self-GLUE vs ID acc | -0.17 | -0.17 | ✓ |
| 4.7c | self-GLUE vs C100 | 0.908 | 0.9078 | ✓ |
| 4.7c | self-GLUE vs Tiny | 0.919 | 0.9192 | ✓ |
| 4.7c | PR vs ID acc | -0.074 | -0.0741 | ✓ |
| 4.7c | PR vs C100 | 0.951 | 0.9507 | ✓ |
| 4.7c | PR vs Tiny | 0.945 | 0.9448 | ✓ |
| 4.7c | numrank vs ID acc | -0.044 | -0.0436 | ✓ |
| 4.7c | numrank vs C100 | 0.944 | 0.9443 | ✓ |
| 4.7c | numrank vs Tiny | 0.949 | 0.9487 | ✓ |
| 4.7c | smoothness vs ID acc | -0.09 | -0.0903 | ✓ |
| 4.7c | smoothness vs C100 | 0.899 | 0.8987 | ✓ |
| 4.7c | smoothness vs Tiny | 0.878 | 0.8778 | ✓ |
| 4.7c | ID acc vs C100 | -0.045 | -0.0453 | ✓ |
| 4.7c | ID acc vs Tiny | -0.027 | -0.0272 | ✓ |
| 4.6 | fog ID acc pooled rho | 0.987 | 0.9871 | ✓ |
| 4.6 | fog ID acc resid rho | 0.918 | 0.918 | ✓ |
| 4.6 | brightness ID acc pooled rho | 0.994 | 0.9944 | ✓ |
| 4.6 | brightness ID acc resid rho | 0.968 | 0.9679 | ✓ |
| 4.6 | contrast ID acc pooled rho | 0.965 | 0.9646 | ✓ |
| 4.6 | contrast ID acc resid rho | 0.772 | 0.7718 | ✓ |
| 4.6 | gaussian_noise ID acc pooled rho | 0.182 | 0.1817 | ✓ |
| 4.6 | gaussian_noise ID acc resid rho | -0.087 | -0.0871 | ✓ |
| 4.6 | pixelate ID acc pooled rho | -0.122 | -0.1222 | ✓ |
| 4.6 | pixelate ID acc resid rho | 0.215 | 0.2155 | ✓ |
| 4.6 | jpeg_compression ID acc pooled rho | 0.122 | 0.1224 | ✓ |
| 4.6 | jpeg_compression ID acc resid rho | -0.01 | -0.0105 | ✓ |
| 4.7a | healthy model count | 79 | 79 | ✓ |
| 4.5 | 224 D_eff bootstrap CI | [+0.879,+0.955] | [+0.879,+0.955] | ✓ |
| 4.5 | 224 D_eff bootstrap robust rate | 1.00 | 1.0000 | ✓ |
| 4.5 | 224 Psi_eff bootstrap CI | [+0.842,+0.938] | [+0.842,+0.938] | ✓ |
| 4.5 | 224 Psi_eff bootstrap robust rate | 1.00 | 1.0000 | ✓ |
| 4.5 | 224 ID accuracy bootstrap CI | [+0.215,+0.622] | [+0.215,+0.622] | ✓ |
| 4.5 | 224 ID accuracy bootstrap robust rate | 0.13 | 0.1330 | ✓ |
| 4.5 | 80 D_eff bootstrap CI | [+0.581,+0.833] | [+0.581,+0.833] | ✓ |
| 4.5 | 80 D_eff bootstrap robust rate | 1.00 | 0.9980 | ✓ |
| 4.5 | 80 ID accuracy bootstrap CI | [-0.825,-0.417] | [-0.825,-0.417] | ✓ |
| 4.5 | 80 ID accuracy bootstrap robust rate | 0.80 | 0.7970 | ✓ |
| 4.5 | ours(48) num.rank bootstrap CI | [+0.797,+0.923] | [+0.797,+0.923] | ✓ |
| 4.5 | ours(48) num.rank bootstrap robust rate | 1.00 | 0.9995 | ✓ |
| 4.5 | ours(48) ID accuracy bootstrap CI | [-0.659,-0.181] | [-0.659,-0.181] | ✓ |
| 4.5 | ours(48) ID accuracy bootstrap robust rate | 0.25 | 0.2510 | ✓ |
| 4.5 | ours(36) num.rank bootstrap CI | [+0.819,+0.943] | [+0.819,+0.943] | ✓ |
| 4.5 | ours(36) num.rank bootstrap robust rate | 1.00 | 1.0000 | ✓ |
| 4.5 | ours(36) ID accuracy bootstrap CI | [-0.803,-0.420] | [-0.803,-0.420] | ✓ |
| 4.5 | ours(36) ID accuracy bootstrap robust rate | 0.83 | 0.8290 | ✓ |
| 4.7b | own GLUE vs C100 (r) | 0.91 | 0.9078 | ✓ |
| 4.7b | own GLUE vs C100 (rho) | 0.92 | 0.9241 | ✓ |
| 4.7b | official dimension vs OOD, ResNet18 (r) | 0.87 | 0.8652 | ✓ |
| 4.7b | official dimension vs OOD, ResNet18 (rho) | 0.87 | 0.8657 | ✓ |
| 4.7b | official ResNet18 matched n | 41 | 41 | ✓ |
| 4.7a | PR C100 | 0.941 | 0.9405 | ✓ |
| 4.7a | PR Tiny | 0.948 | 0.9481 | ✓ |
| 4.7a | smoothness C100 | 0.956 | 0.9556 | ✓ |
| 4.7a | smoothness Tiny | 0.93 | 0.9302 | ✓ |
| 4.7a | numrank C100 | 0.943 | 0.9426 | ✓ |
| 4.7a | numrank Tiny | 0.952 | 0.9519 | ✓ |
| 4.7a | ID acc C100 | 0.153 | 0.1527 | ✓ |
| 4.7a | ID acc Tiny | 0.145 | 0.1447 | ✓ |
| 4.7a | cross-arch subset n | 36 | 36 | ✓ |
| 4.6-self | corruption scenario count | 19 | 19 | ✓ |
| 4.6-self | scenarios where ID acc ranks 1st | 11 | 11 | ✓ |
| 4.6-self | noise scenario ranks | gaussian_noise=8/impulse_noise=7/shot_noise=8/speckle_noise=7 | gaussian_noise=8/impulse_noise=7/shot_noise=8/speckle_noise=7 | ✓ |
| 4.6-self | contrast ID acc rank (self-built) | 7 | 7 | ✓ |
| 4.6-self | contrast ID acc rho (self-built) | 0.087 | 0.0872 | ✓ |
| 3.2 | sensitivity band set (executed) | 0.5/1/2/3/5 | 0.5/1/2/3/5 | ✓ |
| 3.2 | b=8 is degenerate on 224 (n) | 223 | 223 | ✓ |
| 4.7a | within-arch ResNet18 PR C100 | 0.701 | 0.7009 | ✓ |
| 4.7a | within-arch ResNet18 PR Tiny | 0.707 | 0.7068 | ✓ |
| 4.7a | within-arch ResNet18 ID acc C100 | -0.439 | -0.4392 | ✓ |
| 4.7a | within-arch ResNet18 ID acc Tiny | -0.338 | -0.3377 | ✓ |
| 4.7a | within-arch ShuffleNetV2 PR C100 | 0.988 | 0.9877 | ✓ |
| 4.7a | within-arch ShuffleNetV2 PR Tiny | 0.985 | 0.9854 | ✓ |
| 4.7a | within-arch ShuffleNetV2 ID acc C100 | -0.637 | -0.637 | ✓ |
| 4.7a | within-arch ShuffleNetV2 ID acc Tiny | -0.584 | -0.5844 | ✓ |
| 4.7a | within-arch RegNetY PR C100 | 0.659 | 0.6591 | ✓ |
| 4.7a | within-arch RegNetY PR Tiny | 0.536 | 0.5362 | ✓ |
| 4.7a | within-arch RegNetY ID acc C100 | 0.053 | 0.0531 | ✓ |
| 4.7a | within-arch RegNetY ID acc Tiny | -0.019 | -0.019 | ✓ |
| 4.7a | cross-arch PR rho C100 | 0.982 | 0.9822 | ✓ |
| 4.7a | cross-arch PR rho Tiny | 0.981 | 0.981 | ✓ |
| 4.7a | cross-arch smoothness rho C100 | 0.971 | 0.9714 | ✓ |
| 4.7a | cross-arch smoothness rho Tiny | 0.961 | 0.9614 | ✓ |
| 4.7a | cross-arch numrank rho C100 | 0.967 | 0.9668 | ✓ |
| 4.7a | cross-arch numrank rho Tiny | 0.966 | 0.9656 | ✓ |
| 4.7c | partial r controlling ID acc: self-GLUE | 0.686 | 0.6862 | ✓ |
| 4.7c | partial r controlling ID acc: PR | 0.756 | 0.756 | ✓ |
| 4.7c | partial r controlling ID acc: numrank | 0.849 | 0.849 | ✓ |
| 4.7c | partial r controlling ID acc: smoothness | 0.861 | 0.8613 | ✓ |
| provenance | 224/80 release model-key overlap | 0 (disjoint) | 0 | ✓ |
| Table4 | fog pooled best marker (rho) | acc_test | acc_test | ✓ |
| Table4 | fog resid best marker (rho) | acc_test | acc_test | ✓ |
| Table4 | fog pooled rho of best | 0.99 | 0.9897 | ✓ |
| Table4 | fog ID-acc pooled rho | 0.99 | 0.9897 | ✓ |
| Table4 | fog ID-acc pooled rank | 1 | 1 | ✓ |
| Table4 | fog resid rho of best | 0.934 | 0.934 | ✓ |
| Table4 | fog ID-acc resid rho | 0.934 | 0.934 | ✓ |
| Table4 | fog ID-acc resid rank | 1 | 1 | ✓ |
| Table4 | fog n matched | 43 | 43 | ✓ |
| Table4 | fog pooled best rho (recomputed from pkl) | 0.99 | 0.9897 | ✓ |
| Table4 | fog ID-acc pooled rho (recomputed from pkl) | 0.99 | 0.9897 | ✓ |
| Table4 | fog resid best rho (recomputed from pkl) | 0.934 | 0.934 | ✓ |
| Table4 | fog ID-acc resid rho (recomputed from pkl) | 0.934 | 0.934 | ✓ |
| Table4 | fog pooled best marker (recomputed from pkl) | acc_test | acc_test | ✓ |
| Table4 | fog ID-acc pooled rank (recomputed from pkl) | 1 | 1 | ✓ |
| Table4 | fog resid best marker (recomputed from pkl) | acc_test | acc_test | ✓ |
| Table4 | fog ID-acc resid rank (recomputed from pkl) | 1 | 1 | ✓ |
| Table4 | fog marker count | 12 | 12 | ✓ |
| Table4 | brightness pooled best marker (rho) | acc_test | acc_test | ✓ |
| Table4 | brightness resid best marker (rho) | acc_test | acc_test | ✓ |
| Table4 | brightness pooled rho of best | 0.994 | 0.9938 | ✓ |
| Table4 | brightness ID-acc pooled rho | 0.994 | 0.9938 | ✓ |
| Table4 | brightness ID-acc pooled rank | 1 | 1 | ✓ |
| Table4 | brightness resid rho of best | 0.973 | 0.9727 | ✓ |
| Table4 | brightness ID-acc resid rho | 0.973 | 0.9727 | ✓ |
| Table4 | brightness ID-acc resid rank | 1 | 1 | ✓ |
| Table4 | brightness n matched | 43 | 43 | ✓ |
| Table4 | brightness pooled best rho (recomputed from pkl) | 0.994 | 0.9938 | ✓ |
| Table4 | brightness ID-acc pooled rho (recomputed from pkl) | 0.994 | 0.9938 | ✓ |
| Table4 | brightness resid best rho (recomputed from pkl) | 0.973 | 0.9727 | ✓ |
| Table4 | brightness ID-acc resid rho (recomputed from pkl) | 0.973 | 0.9727 | ✓ |
| Table4 | brightness pooled best marker (recomputed from pkl) | acc_test | acc_test | ✓ |
| Table4 | brightness ID-acc pooled rank (recomputed from pkl) | 1 | 1 | ✓ |
| Table4 | brightness resid best marker (recomputed from pkl) | acc_test | acc_test | ✓ |
| Table4 | brightness ID-acc resid rank (recomputed from pkl) | 1 | 1 | ✓ |
| Table4 | brightness marker count | 12 | 12 | ✓ |
| Table4 | contrast pooled best marker (rho) | acc_test | acc_test | ✓ |
| Table4 | contrast resid best marker (rho) | acc_test | acc_test | ✓ |
| Table4 | contrast pooled rho of best | 0.975 | 0.9745 | ✓ |
| Table4 | contrast ID-acc pooled rho | 0.975 | 0.9745 | ✓ |
| Table4 | contrast ID-acc pooled rank | 1 | 1 | ✓ |
| Table4 | contrast resid rho of best | 0.816 | 0.8164 | ✓ |
| Table4 | contrast ID-acc resid rho | 0.816 | 0.8164 | ✓ |
| Table4 | contrast ID-acc resid rank | 1 | 1 | ✓ |
| Table4 | contrast n matched | 43 | 43 | ✓ |
| Table4 | contrast pooled best rho (recomputed from pkl) | 0.975 | 0.9745 | ✓ |
| Table4 | contrast ID-acc pooled rho (recomputed from pkl) | 0.975 | 0.9745 | ✓ |
| Table4 | contrast resid best rho (recomputed from pkl) | 0.816 | 0.8164 | ✓ |
| Table4 | contrast ID-acc resid rho (recomputed from pkl) | 0.816 | 0.8164 | ✓ |
| Table4 | contrast pooled best marker (recomputed from pkl) | acc_test | acc_test | ✓ |
| Table4 | contrast ID-acc pooled rank (recomputed from pkl) | 1 | 1 | ✓ |
| Table4 | contrast resid best marker (recomputed from pkl) | acc_test | acc_test | ✓ |
| Table4 | contrast ID-acc resid rank (recomputed from pkl) | 1 | 1 | ✓ |
| Table4 | contrast marker count | 12 | 12 | ✓ |
| Table4 | gaussian_noise pooled best marker (rho) | mean_angle_id | mean_angle_id | ✓ |
| Table4 | gaussian_noise resid best marker (rho) | sparsity_id | sparsity_id | ✓ |
| Table4 | gaussian_noise pooled rho of best | -0.632 | -0.632 | ✓ |
| Table4 | gaussian_noise ID-acc pooled rho | 0.185 | 0.1851 | ✓ |
| Table4 | gaussian_noise ID-acc pooled rank | 6 | 6 | ✓ |
| Table4 | gaussian_noise resid rho of best | 0.299 | 0.2988 | ✓ |
| Table4 | gaussian_noise ID-acc resid rho | -0.109 | -0.1092 | ✓ |
| Table4 | gaussian_noise ID-acc resid rank | 6 | 6 | ✓ |
| Table4 | gaussian_noise n matched | 43 | 43 | ✓ |
| Table4 | gaussian_noise pooled best rho (recomputed from pkl) | -0.632 | -0.632 | ✓ |
| Table4 | gaussian_noise ID-acc pooled rho (recomputed from pkl) | 0.185 | 0.1851 | ✓ |
| Table4 | gaussian_noise resid best rho (recomputed from pkl) | 0.299 | 0.2988 | ✓ |
| Table4 | gaussian_noise ID-acc resid rho (recomputed from pkl) | -0.109 | -0.1092 | ✓ |
| Table4 | gaussian_noise pooled best marker (recomputed from pkl) | mean_angle_id | mean_angle_id | ✓ |
| Table4 | gaussian_noise ID-acc pooled rank (recomputed from pkl) | 6 | 6 | ✓ |
| Table4 | gaussian_noise resid best marker (recomputed from pkl) | sparsity_id | sparsity_id | ✓ |
| Table4 | gaussian_noise ID-acc resid rank (recomputed from pkl) | 6 | 6 | ✓ |
| Table4 | gaussian_noise marker count | 12 | 12 | ✓ |
| Table4 | pixelate pooled best marker (rho) | center_alignment_id | center_alignment_id | ✓ |
| Table4 | pixelate resid best marker (rho) | radius_id | radius_id | ✓ |
| Table4 | pixelate pooled rho of best | -0.445 | -0.4454 | ✓ |
| Table4 | pixelate ID-acc pooled rho | -0.128 | -0.128 | ✓ |
| Table4 | pixelate ID-acc pooled rank | 10 | 10 | ✓ |
| Table4 | pixelate resid rho of best | -0.3 | -0.3004 | ✓ |
| Table4 | pixelate ID-acc resid rho | 0.191 | 0.1912 | ✓ |
| Table4 | pixelate ID-acc resid rank | 2 | 2 | ✓ |
| Table4 | pixelate n matched | 43 | 43 | ✓ |
| Table4 | pixelate pooled best rho (recomputed from pkl) | -0.445 | -0.4454 | ✓ |
| Table4 | pixelate ID-acc pooled rho (recomputed from pkl) | -0.128 | -0.128 | ✓ |
| Table4 | pixelate resid best rho (recomputed from pkl) | -0.3 | -0.3004 | ✓ |
| Table4 | pixelate ID-acc resid rho (recomputed from pkl) | 0.191 | 0.1912 | ✓ |
| Table4 | pixelate pooled best marker (recomputed from pkl) | center_alignment_id | center_alignment_id | ✓ |
| Table4 | pixelate ID-acc pooled rank (recomputed from pkl) | 10 | 10 | ✓ |
| Table4 | pixelate resid best marker (recomputed from pkl) | radius_id | radius_id | ✓ |
| Table4 | pixelate ID-acc resid rank (recomputed from pkl) | 2 | 2 | ✓ |
| Table4 | pixelate marker count | 12 | 12 | ✓ |
| Table4 | jpeg_compression pooled best marker (rho) | sparsity_id | sparsity_id | ✓ |
| Table4 | jpeg_compression resid best marker (rho) | utility_id | utility_id | ✓ |
| Table4 | jpeg_compression pooled rho of best | -0.351 | -0.3511 | ✓ |
| Table4 | jpeg_compression ID-acc pooled rho | 0.087 | 0.0869 | ✓ |
| Table4 | jpeg_compression ID-acc pooled rank | 11 | 11 | ✓ |
| Table4 | jpeg_compression resid rho of best | 0.262 | 0.2619 | ✓ |
| Table4 | jpeg_compression ID-acc resid rho | -0.055 | -0.055 | ✓ |
| Table4 | jpeg_compression ID-acc resid rank | 7 | 7 | ✓ |
| Table4 | jpeg_compression n matched | 43 | 43 | ✓ |
| Table4 | jpeg_compression pooled best rho (recomputed from pkl) | -0.351 | -0.3511 | ✓ |
| Table4 | jpeg_compression ID-acc pooled rho (recomputed from pkl) | 0.087 | 0.0869 | ✓ |
| Table4 | jpeg_compression resid best rho (recomputed from pkl) | 0.262 | 0.2619 | ✓ |
| Table4 | jpeg_compression ID-acc resid rho (recomputed from pkl) | -0.055 | -0.055 | ✓ |
| Table4 | jpeg_compression pooled best marker (recomputed from pkl) | sparsity_id | sparsity_id | ✓ |
| Table4 | jpeg_compression ID-acc pooled rank (recomputed from pkl) | 11 | 11 | ✓ |
| Table4 | jpeg_compression resid best marker (recomputed from pkl) | utility_id | utility_id | ✓ |
| Table4 | jpeg_compression ID-acc resid rank (recomputed from pkl) | 7 | 7 | ✓ |
| Table4 | jpeg_compression marker count | 12 | 12 | ✓ |
| Table4-prose | Protocol-A rho range, photometric (min) | 0.816 | 0.8164 | ✓ |
| Table4-prose | Protocol-A rho range, photometric (max) | 0.973 | 0.9727 | ✓ |
| Table4-prose | Protocol-A rho range, noise/compression (min) | -0.109 | -0.1092 | ✓ |
| Table4-prose | Protocol-A rho range, noise/compression (max) | 0.191 | 0.1912 | ✓ |
| Table4-prose | Protocol-A ID-acc rank span, noise/compression (min) | 2 | 2 | ✓ |
| Table4-prose | Protocol-A ID-acc rank span, noise/compression (max) | 7 | 7 | ✓ |
| Table4-prose | Protocol-A ID-acc rank, photometric | 1/1/1 | 1/1/1 | ✓ |
| Table4-prose | pooled ID-acc rho range, photometric (min) | 0.975 | 0.9745 | ✓ |
| Table4-prose | pooled ID-acc rho range, photometric (max) | 0.994 | 0.9938 | ✓ |
| Table4-prose | pixelate ID-acc resid rank | 2 | 2 | ✓ |
| Table4-prose | pixelate ID-acc pooled rho (negative) | -0.128 | -0.128 | ✓ |
| Table4-prose | pixelate ID-acc resid rho (positive) | 0.191 | 0.1912 | ✓ |
| Table4-prose | gaussian noise Protocol-A best marker | sparsity_id | sparsity_id | ✓ |
| Table4-prose | pixelate Protocol-A best marker | radius_id | radius_id | ✓ |
| Table4-prose | jpeg Protocol-A best marker | utility_id | utility_id | ✓ |
| Table4-prose | gaussian noise Protocol-A best rho | 0.299 | 0.2988 | ✓ |
| Table4-prose | pixelate Protocol-A best rho | -0.3 | -0.3004 | ✓ |
| Table4-prose | jpeg Protocol-A best rho | 0.262 | 0.2619 | ✓ |
| Table4-prose | gaussian_noise Protocol-A best/second gap | 0.0699 | 0.0699 | ✓ |
| Table4-prose | pixelate Protocol-A best/second gap | 0.1092 | 0.1092 | ✓ |
| Table4-prose | jpeg_compression Protocol-A best/second gap | 0.0306 | 0.0306 | ✓ |
| Table4-prose | scenarios with best/second gap < 0.04 (Protocol A) | 1 | 1 | ✓ |
| Table4-prose | scenarios with best/second gap < 0.04 (pooled) | 1 | 1 | ✓ |
| Table4-prose | jpeg best/second gap < 0.04 (Protocol A) | 1 | 1 | ✓ |
| signmech | 224-release n | 106 | 106 | ✓ |
| signmech | 224-release ID~lr | -0.451 | -0.4506 | ✓ |
| signmech | 224-release OOD~lr | -0.815 | -0.8147 | ✓ |
| signmech | 224-release r, lr blocked | 0.043 | 0.0433 | ✓ |
| signmech | 224-release r, wd blocked (placebo) | 0.391 | 0.3908 | ✓ |
| signmech | 80-release n | 43 | 43 | ✓ |
| signmech | 80-release ID~lr | 0.089 | 0.089 | ✓ |
| signmech | 80-release OOD~lr | -0.487 | -0.4871 | ✓ |
| signmech | 80-release r, lr blocked | -0.67 | -0.6696 | ✓ |
| signmech | 80-release r, wd blocked (placebo) | -0.614 | -0.614 | ✓ |
| signmech | 224-release Protocol-A r before blocking | 0.392 | 0.3922 | ✓ |
| signmech | 80-release Protocol-A r before blocking | -0.628 | -0.6281 | ✓ |
| signmech | 224-release b=2 before blocking | 0.57 | 0.57 | ✓ |
| signmech | 224-release b=2 after lr blocking | 0.23 | 0.2263 | ✓ |
| signmech | 224-release b=3 before blocking | 0.57 | 0.5663 | ✓ |
| signmech | 224-release b=3 after lr blocking | 0.09 | 0.0949 | ✓ |
| signmech | 80-release b=3 before blocking | -0.7 | -0.6988 | ✓ |
| signmech | 80-release b=3 after lr blocking | -0.73 | -0.7312 | ✓ |
| glueown | matched n (all 48) | 45 | 45 | ✓ |
| glueown | matched n (converged) | 34 | 34 | ✓ |
| glueown | D_eff resid r (all 48) | 0.698 | 0.6975 | ✓ |
| glueown | D_eff rho (all 48) | 0.64 | 0.6402 | ✓ |
| glueown | D_eff C4 sign (all 48) | fail | fail | ✓ |
| glueown | Psi_eff resid r (all 48) | 0.098 | 0.0983 | ✓ |
| glueown | Psi_eff rho (all 48) | 0.276 | 0.2758 | ✓ |
| glueown | Psi_eff C4 sign (all 48) | fail | fail | ✓ |
| glueown | R_eff resid r (all 48) | 0.462 | 0.4621 | ✓ |
| glueown | R_eff rho (all 48) | 0.484 | 0.4842 | ✓ |
| glueown | R_eff C4 sign (all 48) | pass | pass | ✓ |
| glueown | ID acc resid r (all 48) | -0.453 | -0.4533 | ✓ |
| glueown | ID acc rho (all 48) | -0.428 | -0.4283 | ✓ |
| glueown | ID acc C4 sign (all 48) | fail | fail | ✓ |
| glueown | D_eff resid r (converged) | 0.714 | 0.7136 | ✓ |
| glueown | D_eff rho (converged) | 0.667 | 0.6669 | ✓ |
| glueown | D_eff C4 sign (converged) | fail | fail | ✓ |
| glueown | Psi_eff resid r (converged) | 0.654 | 0.6538 | ✓ |
| glueown | Psi_eff rho (converged) | 0.632 | 0.6315 | ✓ |
| glueown | Psi_eff C4 sign (converged) | pass | pass | ✓ |
| glueown | R_eff resid r (converged) | 0.659 | 0.659 | ✓ |
| glueown | R_eff rho (converged) | 0.675 | 0.6752 | ✓ |
| glueown | R_eff C4 sign (converged) | pass | pass | ✓ |
| glueown | ID acc resid r (converged) | -0.658 | -0.6577 | ✓ |
| glueown | ID acc rho (converged) | -0.635 | -0.6348 | ✓ |
| glueown | ID acc C4 sign (converged) | pass | pass | ✓ |
| glueown | Psi_eff per-family signs (all 48) | +1/+1/+1/-1 | +1/+1/+1/-1 | ✓ |
| glueown | ID-acc per-family signs (all 48) | -1/-1/-1/+1 | -1/-1/-1/+1 | ✓ |
| glueown | ID-acc per-family signs (converged) | -1/-1/-1 | -1/-1/-1 | ✓ |
| glueown | D_eff lr=0.25 all-family r | 0.8 | 0.7958 | ✓ |
| glueown | D_eff lr=0.25 matched-subset r | -0.41 | -0.4074 | ✓ |
| glueown | D_eff lr=0.25 matched n | 11 | 11 | ✓ |
| glueown | D_eff pooled-center (no lr resid) n | 32 | 32 | ✓ |
| glueown | D_eff pooled-center (no lr resid) r | 0.91 | 0.9194 | ✓ |
| bootstrap | 224 PR bootstrap robust rate | 0.5535 | 0.5535 | ✓ |
| bootstrap | 224 D_eff bootstrap robust rate | 1.0 | 1.0 | ✓ |
| 4.7a | trained model count | 80 | 80 | ✓ |
| 4.7a | diverged model count | 1 | 1 | ✓ |
| 4.7a | healthy model count (recomputed) | 79 | 79 | ✓ |
| 4.7a | resnet18 trained models | 48 | 48 | ✓ |
| 4.7a | resnet18 healthy models | 48 | 48 | ✓ |
| 4.7a | shufflenet_v2_x1_0 trained models | 16 | 16 | ✓ |
| 4.7a | shufflenet_v2_x1_0 healthy models | 15 | 15 | ✓ |
| 4.7a | regnet_y_400mf trained models | 16 | 16 | ✓ |
| 4.7a | regnet_y_400mf healthy models | 16 | 16 | ✓ |
| 4.7a | cross-arch subset n (ShuffleNetV2 excluded) | 36 | 36 | ✓ |
| 4.7a | ShuffleNetV2 models inside pooled band | 0 | 0 | ✓ |
| 4.7a | pooled band lower bound (pp) | 83.192 | 83.1922 | ✓ |
| 4.7a | pooled band upper bound (pp) | 87.192 | 87.1922 | ✓ |
| 4.7a | ShuffleNetV2 healthy model count | 15 | 15 | ✓ |
| 4.7a | ShuffleNetV2 ID-accuracy ceiling (pp) | 81.91 | 81.91 | ✓ |
| 4.7a | ShuffleNetV2 ceiling below band lower bound | 1 | 1 | ✓ |
| 4.7a | within-arch ShuffleNetV2 PR C100 | 0.988 | 0.9877 | ✓ |
| 4.7a | within-arch ShuffleNetV2 PR n | 14 | 14 | ✓ |
| 4.7a | within-arch ShuffleNetV2 smoothness C100 | 0.993 | 0.9929 | ✓ |
| 4.7a | within-arch ShuffleNetV2 smoothness n | 14 | 14 | ✓ |
| 4.7a | within-arch ShuffleNetV2 numrank C100 | 0.306 | 0.306 | ✓ |
| 4.7a | within-arch ShuffleNetV2 numrank n | 14 | 14 | ✓ |
| 4.7a | within-arch ShuffleNetV2 ID acc C100 | -0.637 | -0.637 | ✓ |
| 4.7a | within-arch ShuffleNetV2 ID acc n | 14 | 14 | ✓ |
| 4.7a | within-arch RegNetY n | 13 | 13 | ✓ |
| 4.7a | within-arch RegNetY PR C100 | 0.659 | 0.6591 | ✓ |
| 4.7a | within-arch RegNetY ID acc C100 | 0.053 | 0.0531 | ✓ |
| 4.7a | within-arch RegNetY ID acc Tiny | -0.019 | -0.019 | ✓ |
| 4.7a | within-arch RegNetY PR Tiny | 0.536 | 0.5362 | ✓ |
| appendixA | D_eff resid rho (80 release) | 0.81 | 0.8052 | ✓ |
| appendixA | Psi_eff resid rho (80 release) | 0.8 | 0.799 | ✓ |
| appendixA | ID acc resid rho (80 release) | -0.61 | -0.6059 | ✓ |
| appendixA | PR resid rho (80 release) | 0.79 | 0.7936 | ✓ |
| appendixA | mean_dist (avg) resid rho (80 release) | 0.68 | 0.6805 | ✓ |
| appendixA | raw pooled r, 80 release | 0.4426 | 0.4426 | ✓ |
| appendixA | raw pooled r, 224 release | 0.4759 | 0.4759 | ✓ |
| setup | 224 union learning rates | 13 | 13 | ✓ |
| setup | 224 union weight decays | 9 | 9 | ✓ |
| setup | 224 architecture count | 5 | 5 | ✓ |
| setup | 80 release scenario count | 6 | 6 | ✓ |
| setup | 80 release model count | 80 | 80 | ✓ |
| text-table4 | tab:shifttype row fog | ID accuracy | +0.990 | +0.990 | 1/12 | ID accuracy | +0.934 | +0.934 | 1/12 | ID accuracy | +0.990 | +0.990 | 1/12 | ID accuracy | +0.934 | +0.934 | 1/12 | ✓ |
| text-table4 | tab:shifttype row brightness | ID accuracy | +0.994 | +0.994 | 1/12 | ID accuracy | +0.973 | +0.973 | 1/12 | ID accuracy | +0.994 | +0.994 | 1/12 | ID accuracy | +0.973 | +0.973 | 1/12 | ✓ |
| text-table4 | tab:shifttype row contrast | ID accuracy | +0.975 | +0.975 | 1/12 | ID accuracy | +0.816 | +0.816 | 1/12 | ID accuracy | +0.975 | +0.975 | 1/12 | ID accuracy | +0.816 | +0.816 | 1/12 | ✓ |
| text-table4 | tab:shifttype row gaussian_noise | mean angle | -0.632 | +0.185 | 6/12 | sparsity | +0.299 | -0.109 | 6/12 | mean angle | -0.632 | +0.185 | 6/12 | sparsity | +0.299 | -0.109 | 6/12 | ✓ |
| text-table4 | tab:shifttype row pixelate | center alignment | -0.445 | -0.128 | 10/12 | effective radius | -0.300 | +0.191 | 2/12 | center alignment | -0.445 | -0.128 | 10/12 | effective radius | -0.300 | +0.191 | 2/12 | ✓ |
| text-table4 | tab:shifttype row jpeg_compression | sparsity | -0.351 | +0.087 | 11/12 | effective utility | +0.262 | -0.055 | 7/12 | sparsity | -0.351 | +0.087 | 11/12 | effective utility | +0.262 | -0.055 | 7/12 | ✓ |
| text-signmech | tab:signmech row 224 models | 106 | -0.451 | -0.815 | +0.043 | 106 | -0.451 | -0.815 | +0.043 | ✓ |
| text-signmech | tab:signmech row 80 models | 43 | +0.089 | -0.487 | -0.670 | 43 | +0.089 | -0.487 | -0.670 | ✓ |
| text-glueown | tab:glueown row D_eff (own impl.) | +0.698 | +0.640 | fail | +0.698 | +0.640 | fail | ✓ |
| text-glueown | tab:glueown row Psi_eff (own impl.) | +0.098 | +0.276 | fail | +0.098 | +0.276 | fail | ✓ |
| text-glueown | tab:glueown row R_eff (own impl.) | +0.462 | +0.484 | pass | +0.462 | +0.484 | pass | ✓ |
| text-glueown | tab:glueown row ID accuracy | -0.453 | -0.428 | fail | -0.453 | -0.428 | fail | ✓ |
| text-table3 | tab:bootstrap robust rate, ours (48, all) | 25% | 25% | ✓ |
| text-table3 | tab:bootstrap robust rate, ours (36, high-transfer) | 83% | 83% | ✓ |
| text | main.tex contains '$-0.11$ to $+0.19$' | present | present | ✓ |
| text | main.tex contains '2nd--7th' | present | present | ✓ |
| text | main.tex contains '55\\%' | present | present | ✓ |
| text | main.tex contains '$r=+0.71$' | present | present | ✓ |
| text | main.tex contains '$r=+0.65$' | present | present | ✓ |
| text | main.tex contains '$r=+0.66$' | present | present | ✓ |
| text | main.tex contains 'r=-0.66' | present | present | ✓ |
| text | main.tex contains '79 healthy models out of 80 trained' | present | present | ✓ |
| text | main.tex contains 'none of its 15 healthy models' | present | present | ✓ |
| text | main.tex contains '$r=+0.988$' | present | present | ✓ |
| text | main.tex contains '$r=+0.993$' | present | present | ✓ |
| text | main.tex contains '$r=+0.306$' | present | present | ✓ |
| text | main.tex contains '$-0.64$' | present | present | ✓ |
| text | main.tex contains '($n=13$)' | present | present | ✓ |
| text | main.tex contains '$n=45$' | present | present | ✓ |
| text | main.tex contains 'pooled $r=+0.44$' | present | present | ✓ |
| text | main.tex contains 'mean\\_dist (avg)' | present | present | ✓ |
| text | main.tex must not contain '+0.918' | absent | absent | ✓ |
| text | main.tex must not contain '+0.967$ & $+0.967$ & 1/12' | absent | absent | ✓ |
| text | main.tex must not contain '-0.087$ & $-0.087$' | absent | absent | ✓ |
| text | main.tex must not contain '23\\%' | absent | absent | ✓ |
| text | main.tex must not contain '54\\%' | absent | absent | ✓ |
| text | main.tex must not contain '78 healthy' | absent | absent | ✓ |
| text | main.tex must not contain '78 self-trained' | absent | absent | ✓ |
| text | main.tex must not contain 'magnitude collapse' | absent | absent | ✓ |
| text | main.tex: 80-release pooled r is +0.44, not +0.48 | +0.44 | +0.44 | ✓ |
