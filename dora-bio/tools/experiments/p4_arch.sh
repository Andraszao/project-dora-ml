#!/usr/bin/env bash
# P4 brick 21: dilated CNN vs U-Net on real FMD data (results in docs/RESULTS.md).
#
#   tools/fetch_fmd.sh && tools/experiments/p3_struct.sh && tools/experiments/p4_arch.sh
#
# The dilated baselines are P3's runs (runs/p3/sup_plain.bin, runs/p3/n2v_plain.bin). The U-Net (16 → 32 → 64
# features, depth 2) gets exactly their budget, seed, data and validation: 30 epochs × 50 steps × batch 16 × 64 px,
# seed 2026, 4 raw frames per FOV, FOV 19 never used. Then both are timed on a 512×512 frame. The rule for the
# default was set before running (PATHWAY.md, P4): the U-Net wins only if it is better for supervised AND
# Noise2Void training (all-48 PSNR, raw) and at most 3× slower. Finished runs and evaluations are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/p4 BASE=runs/p3
mkdir -p "$OUT"
[[ -f "$BASE/sup_plain.bin" && -f "$BASE/n2v_plain.bin" ]] || { echo "run tools/experiments/p3_struct.sh first"; exit 1; }
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
COMMON=(--data data/fmd --epochs 30 --steps 50 --batch 16 --patch 64 --seed 2026 --fmd-frames 4 --fmd-levels raw
        --journey "$OUT/journey.json" --arch unet --features 16 --depth 2)

train() {   # train <name> <description> <train args...>
  local name="$1" description="$2"; shift 2
  [[ -f "$OUT/$name.bin" ]] && return 0
  echo "=== train $name: $description"
  "${CLI[@]}" train "${COMMON[@]}" --out "$OUT/$name.bin" --describe "FMD (12 categories, FOV 19 excluded): $description" "$@" \
    > "$OUT/$name.train.log"
  tail -2 "$OUT/$name.train.log"
}
evaluate() {   # evaluate <name> <weights>
  local name="$1" weights="$2"
  [[ -f "$OUT/$name.eval.txt" ]] || "${CLI[@]}" fmd-eval --weights "$weights" --root data/fmd --levels raw > "$OUT/$name.eval.txt"
  [[ -f "$OUT/$name.levels.txt" ]] || "${CLI[@]}" fmd-eval --weights "$weights" --root data/fmd --levels avg2,avg4,avg8,avg16 > "$OUT/$name.levels.txt"
  echo "--- $name"; cat "$OUT/$name.eval.txt"
}

train unet_sup "U-Net, supervised raw -> ground truth" --mode supervised
train unet_n2v "U-Net, Noise2Void, raw frames, plain blind spots" --mode n2v --struct off

evaluate dncnn_sup "$BASE/sup_plain.bin"
evaluate unet_sup "$OUT/unet_sup.bin"
evaluate dncnn_n2v "$BASE/n2v_plain.bin"
evaluate unet_n2v "$OUT/unet_n2v.bin"

# Speed: denoise one 512×512 test frame 5 times with each model (all cores, tiled); keep the median.
frame=data/fmd/test_mix/raw/Confocal_BPAE_B_1.png
median_seconds() {
  for _ in 1 2 3 4 5; do "${CLI[@]}" denoise --weights "$1" --in "$frame" --out "$OUT/timing.tif" | sed -n 's/.* in \([0-9.]*\)s .*/\1/p'; done \
    | sort -n | sed -n 3p
}
dncnn_s="$(median_seconds "$BASE/sup_plain.bin")"
unet_s="$(median_seconds "$OUT/unet_sup.bin")"
awk -v a="$unet_s" -v b="$dncnn_s" 'BEGIN { printf "512x512 denoise, median of 5: dilated %.3f s, U-Net %.3f s, slowdown %.2f\n", b, a, a / b }' \
  | tee "$OUT/timing.txt"
echo "all P4 architecture runs done"
