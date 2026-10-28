#!/usr/bin/env bash
# Reproduce the full analysis pipeline.
#
#   ./run_all.sh verify    # re-derive all 111 headline numbers from raw data (no GPU, ~1 min)
#   ./run_all.sh analyze   # regenerate every table and report (no GPU, ~10 min)
#   ./run_all.sh figures   # regenerate Figures 2-6 (no GPU, ~2 min)
#   ./run_all.sh all       # verify + analyze + figures
#
# Training and measurement (GPU) are separate; see README section 4.6-4.7.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python}"
MODE="${1:-all}"

have_data () {
  [ -f "data/official/df_fig3.pkl" ] && [ -f "data/self_trained/ind_validation_results.json" ]
}

require_data () {
  if ! have_data; then
    echo "[run_all] ERROR: data/official or data/self_trained is missing." >&2
    echo "          Run 'bash code/fetch_ood.sh' and place the released pickles as:" >&2
    echo "            data/official/df_fig3.pkl" >&2
    echo "            data/official/df_fig15-20.pkl" >&2
    echo "            data/self_trained/ind_validation_results.json" >&2
    echo "            data/self_trained/mod_arch_results.json" >&2
    exit 1
  fi
}

do_verify () {
  echo "=== [1/3] Verifying every reported number against the raw data ==="
  $PY code/audit_claims.py
}

do_analyze () {
  require_data
  echo "=== [2/3] Re-running the analyses ==="
  $PY code/analyze_fig3.py            --outdir results
  $PY code/analyze_independent.py      --outdir results
  $PY code/analyze_multiarch.py        --outdir results
  $PY code/bootstrap_ci.py             --outdir results
  $PY code/analyze_ind48_robust.py     --outdir results
  $PY code/analyze_corruption_dual.py  --outdir results
  echo "--- analyses done ---"
}

do_figures () {
  require_data
  echo "=== [3/3] Regenerating Figures 2-6 ==="
  $PY code/make_figures.py --outdir figures
}

case "$MODE" in
  verify)  do_verify ;;
  analyze) do_analyze ;;
  figures) do_figures ;;
  all)     do_verify; do_analyze; do_figures ;;
  *) echo "usage: $0 {verify|analyze|figures|all}" >&2; exit 2 ;;
esac

echo "[run_all] done ($MODE)"
