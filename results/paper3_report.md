# Paper 3 stats (auto-generated)

## E1: 8 models, 152 rows

base: {'n': 8, 'pearson_r': 0.9681, 'spearman_rho': 0.881, 'slope': 0.1367, 'd_range': [0.89, 2.136]}

- gauss_iso: median -0.00810, neg 8/8, range [-0.0234, -0.006]
- gauss_aniso: median +0.13748, neg 3/8, range [-1.7529, 0.8345]
- recolor: median +0.01052, neg 1/8, range [-0.018, 0.1655]
- tot_recolor: median +0.01090, neg 1/8, range [-0.0, 0.1248]

aniso2: {'dc100_median': -0.1077, 'dc100_all_neg': True, 'didp_median': -0.1265, 'didp_all_neg': True, 'dD_median': 0.0005, 'relD_max': 0.4416, 'per_model_dc100': [-0.1317, -0.1071, -0.1299, -0.1084, -0.1109, -0.0986, -0.1053, -0.0861]}
rotate: {'dc100_max_abs': 0.0172, 'dD_max_abs': 0.2081}
iso2: {'dc100_median': -0.0646, 'dc100_all_neg': True, 'dD_median': 5.774}
recolor0: {'dc100_median': 0.017, 'all_pos': False, 'per_model': [0.0272, 0.0562, 0.0019, 0.0201, 0.0122, 0.0306, 0.014, -0.0003]}
geometric share: {'geo_slope_median': 0.01052, 'obs_slope': 0.1367, 'share_pct': 7.7, 'pooled_geo_slope_median': 0.01361, 'pooled_share_pct': 10.0}

## E3: 8 paired models
{
 "n_models": 8,
 "train_whiten_dc100": {
  "median": 0.017,
  "pos": 7,
  "wilcoxon_p": 0.0156,
  "per_model": [
   0.0272,
   0.0562,
   0.0019,
   0.0201,
   0.0122,
   0.0306,
   0.014,
   -0.0003
  ]
 },
 "inference_recolor_dfc": {
  "median": -0.0013,
  "neg": 7,
  "wilcoxon_p": 0.0156
 },
 "head_original_space": {
  "median": -0.0035,
  "neg": 8,
  "wilcoxon_p": 0.0078
 },
 "id_probe_diff_median": 0.0,
 "lr_strat": {
  "0.125": {
   "n": 3,
   "median": 0.0272,
   "per_model": [
    0.0272,
    0.0562,
    0.0019
   ]
  },
  "0.5": {
   "n": 3,
   "median": 0.0201,
   "per_model": [
    0.0201,
    0.0122,
    0.0306
   ]
  },
  "1.0": {
   "n": 2,
   "median": 0.0068,
   "per_model": [
    0.014,
    -0.0003
   ]
  }
 }
}

## E2:
{
 "resnet34": {
  "status": "complete",
  "n_models": 4,
  "n_cond": 19,
  "base": {
   "n": 4,
   "pearson_r": 0.9771,
   "spearman_rho": 0.8,
   "slope": 0.1332,
   "d_range": [
    1.069,
    1.752
   ]
  },
  "recolor_slope_median": 0.00439,
  "recolor_neg": 1,
  "recolor_share_pct": 3.3,
  "recolor0_all_pos": true,
  "recolor0_median": 0.0187,
  "aniso2_dc100_median": -0.0964,
  "aniso2_all_neg": true
 },
 "resnet50": {
  "status": "complete",
  "n_models": 4,
  "n_cond": 19,
  "base": {
   "n": 4,
   "pearson_r": 0.9848,
   "spearman_rho": 1.0,
   "slope": 0.0869,
   "d_range": [
    0.998,
    3.295
   ]
  },
  "recolor_slope_median": 0.00166,
  "recolor_neg": 1,
  "recolor_share_pct": 1.9,
  "recolor0_all_pos": false,
  "recolor0_median": 0.0111,
  "aniso2_dc100_median": -0.1186,
  "aniso2_all_neg": true
 },
 "shufflenet_v2_x1_0": {
  "status": "complete",
  "n_models": 4,
  "n_cond": 19,
  "base": {
   "n": 4,
   "pearson_r": 0.8949,
   "spearman_rho": 1.0,
   "slope": 0.0701,
   "d_range": [
    1.012,
    3.51
   ]
  },
  "recolor_slope_median": -0.00585,
  "recolor_neg": 4,
  "recolor_share_pct": -8.3,
  "recolor0_all_pos": false,
  "recolor0_median": -0.006,
  "aniso2_dc100_median": -0.1314,
  "aniso2_all_neg": true
 },
 "regnet_y_400mf": {
  "status": "complete",
  "n_models": 4,
  "n_cond": 19,
  "base": {
   "n": 4,
   "pearson_r": 0.8672,
   "spearman_rho": 1.0,
   "slope": 0.1139,
   "d_range": [
    1.031,
    2.488
   ]
  },
  "recolor_slope_median": 0.00681,
  "recolor_neg": 0,
  "recolor_share_pct": 6.0,
  "recolor0_all_pos": true,
  "recolor0_median": 0.0074,
  "aniso2_dc100_median": -0.1595,
  "aniso2_all_neg": true
 }
}
