# 论文数字一致性审计（paper_draft_v1.md vs 原始数据重算）

生成：2026-10-27 · 共 133 条断言 · 通过 133 · 不一致 0

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
| 4.5 | B1_fig3_224/dimension_id bootstrap CI | [+0.879,+0.955] rr=1.00 | [+0.879,+0.955] rr=1.00 | ✓ |
| 4.5 | B1_fig3_224/utility_id bootstrap CI | [+0.842,+0.938] rr=1.00 | [+0.842,+0.938] rr=1.00 | ✓ |
| 4.5 | B2_official_80/dimension_id bootstrap CI | [+0.581,+0.833] rr=1.00 | [+0.581,+0.833] rr=1.00 | ✓ |
| 4.5 | B3_own_48/id_numerical_rank bootstrap CI | [+0.797,+0.923] rr=1.00 | [+0.797,+0.923] rr=1.00 | ✓ |
| 4.5 | B1_fig3_224/acc_test bootstrap CI | [+0.215,+0.622] rr=0.13 | [+0.215,+0.622] rr=0.13 | ✓ |
| 4.5 | B2_official_80/acc_test bootstrap CI | [-0.825,-0.417] rr=0.80 | [-0.825,-0.417] rr=0.80 | ✓ |
| 4.5 | B3_own_48/acc_test bootstrap CI | [-0.659,-0.181] rr=0.23 | [-0.659,-0.181] rr=0.23 | ✓ |
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
