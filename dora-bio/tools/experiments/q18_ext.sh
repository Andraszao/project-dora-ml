#!/usr/bin/env bash
# Quality pass Q18 (brick 31): DORA against CAREamics Noise2Void and CARE on public paired datasets, pre-registered in
# PATHWAY.md (commit be5228b) before any of its code existed. Results in docs/RESULTS.md.
#
#   tools/fetch_ext.sh
#   python3 -m venv VENV && VENV/bin/pip install -r tools/experiments/requirements-careamics.txt
#   CAREAMICS_PYTHON=VENV/bin/python tools/experiments/q18_ext.sh
#
# For each of the three paired sets: DORA Noise2Void (default recipe: dilated network, 30 x 50 steps, on the training
# split's noisy images), DORA supervised (U-Net 16 -> 64, depth 2, 60 x 50 steps, on the training pairs), CAREamics
# Noise2Void and CARE (default 2D configs, the same steps, batch and patch size, on CPU), and DORA out of the box (the
# bundled synthetic model and P4's FMD-trained U-Net). Everything is scored by `dora-bio ext-eval` on the test split.
# Finished runs and evaluations are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/q18
PY="${CAREAMICS_PYTHON:?set CAREAMICS_PYTHON to a python with tools/experiments/requirements-careamics.txt installed}"
mkdir -p "$OUT"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
DATASETS=(deepbacs-mreb deepbacs-nucleoid zerocost-care2d)
COMMON=(--data data/ext --steps 50 --batch 16 --patch 64 --seed 2026 --journey "$OUT/journey.json")

for d in "${DATASETS[@]}"; do
  if [[ ! -f "$OUT/dora_n2v_$d.bin" ]]; then
    echo "=== DORA Noise2Void on $d"
    "${CLI[@]}" train --ext "$d" "${COMMON[@]}" --mode n2v --epochs 30 --out "$OUT/dora_n2v_$d.bin" \
      --describe "$d training split, noisy images only: DORA Noise2Void, default recipe" > "$OUT/dora_n2v_$d.train.log"
    tail -1 "$OUT/dora_n2v_$d.train.log"
  fi
  if [[ ! -f "$OUT/dora_sup_$d.bin" ]]; then
    echo "=== DORA supervised U-Net on $d"
    "${CLI[@]}" train --ext "$d" "${COMMON[@]}" --mode supervised --arch unet --features 16 --depth 2 --epochs 60 \
      --out "$OUT/dora_sup_$d.bin" --describe "$d training pairs: DORA supervised U-Net" > "$OUT/dora_sup_$d.train.log"
    tail -1 "$OUT/dora_sup_$d.train.log"
  fi
done
for d in "${DATASETS[@]}"; do
  for a in n2v care; do
    [[ -f "$OUT/careamics_$a/$d/timing.json" ]] && continue
    echo "=== CAREamics $a on $d"
    "$PY" tools/experiments/ext_careamics.py --algorithm "$a" --dataset "$d" --root data/ext --out "$OUT" > "$OUT/careamics_${a}_$d.log" 2>&1
    tail -1 "$OUT/careamics_${a}_$d.log"
  done
done

evaluate() {   # evaluate <name> <dataset> <ext-eval args...>
  local f="$OUT/eval_$1_$2.txt"; shift 2
  [[ -f "$f" ]] || "${CLI[@]}" ext-eval "$@" > "$f"
}
for d in "${DATASETS[@]}"; do
  evaluate bundled "$d" --dataset "$d" --weights godot/models/dora_bio_weights.bin
  [[ -f runs/p4/unet_sup.bin ]] && evaluate fmd "$d" --dataset "$d" --weights runs/p4/unet_sup.bin
  evaluate dora_n2v "$d" --dataset "$d" --weights "$OUT/dora_n2v_$d.bin"
  evaluate dora_sup "$d" --dataset "$d" --weights "$OUT/dora_sup_$d.bin"
  evaluate careamics_n2v "$d" --dataset "$d" --predictions "$OUT/careamics_n2v/$d" \
    --label "CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU"
  evaluate careamics_care "$d" --dataset "$d" --predictions "$OUT/careamics_care/$d" \
    --label "CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU"
  echo "--- $d (mean PSNR / SSIM: noisy, bilateral, method)"
  for m in bundled fmd dora_n2v careamics_n2v dora_sup careamics_care; do
    [[ -f "$OUT/eval_${m}_$d.txt" ]] && printf "%-16s %s\n" "$m" "$(grep '^mean' "$OUT/eval_${m}_$d.txt" | cut -c14-)"
  done
done
echo "all Q18 runs done"
