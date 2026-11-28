#!/usr/bin/env bash
# Quality pass Q13 (brick 29): a noise-conditioned U-Net (FiLM), pre-registered in PATHWAY.md (commit c14066f) before
# any of its code existed. Results in docs/RESULTS.md.
#
#   tools/fetch_fmd.sh && tools/experiments/p4_arch.sh && tools/experiments/q1_supervised.sh && tools/experiments/q13_conditioning.sh
#
# Candidate: Q1's all-level recipe (U-Net 16 -> 64 features, depth 2, supervised, all five FMD noise levels with 2 frames
# each, up to 60 epochs) with --condition noise. Seeds 2026 (decides the bar), 2027 and 2028, for the candidate and for
# the plain all-level recipe (its seed-2026 run is Q1's runs/q1/unet_levels60.bin). Mechanism check: the seed-2026
# candidate on its own validation FOVs with the condition from 1x, 2x and 0.5x the measured noise σ.
# Finished runs and evaluations are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/q13 PLAIN2026=runs/q1/unet_levels60.bin
mkdir -p "$OUT"
[[ -f "$PLAIN2026" ]] || { echo "run tools/experiments/q1_supervised.sh first"; exit 1; }
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
LEVELS=raw,avg2,avg4,avg8,avg16
COMMON=(--data data/fmd --steps 50 --batch 16 --patch 64 --journey "$OUT/journey.json" --epochs 60
        --mode supervised --arch unet --features 16 --depth 2 --fmd-frames 2 --fmd-levels $LEVELS)

train() {   # train <name> <seed> <condition> <description>
  [[ -f "$OUT/$1.bin" ]] && return 0
  echo "=== train $1: $4"
  "${CLI[@]}" train "${COMMON[@]}" --seed "$2" --condition "$3" --out "$OUT/$1.bin" \
    --describe "FMD (12 categories, FOV 19 excluded): $4" > "$OUT/$1.train.log"
  tail -2 "$OUT/$1.train.log"
}
train cond_s2026 2026 noise "U-Net, noise-conditioned (FiLM), supervised, all 5 noise levels, 60 epochs, seed 2026"
train cond_s2027 2027 noise "U-Net, noise-conditioned (FiLM), supervised, all 5 noise levels, 60 epochs, seed 2027"
train cond_s2028 2028 noise "U-Net, noise-conditioned (FiLM), supervised, all 5 noise levels, 60 epochs, seed 2028"
train plain_s2027 2027 none "U-Net, supervised, all 5 noise levels, 60 epochs, seed 2027"
train plain_s2028 2028 none "U-Net, supervised, all 5 noise levels, 60 epochs, seed 2028"

for name in cond_s2026 cond_s2027 cond_s2028 plain_s2027 plain_s2028; do
  [[ -f "$OUT/$name.eval.txt" ]] || "${CLI[@]}" fmd-eval --weights "$OUT/$name.bin" --root data/fmd --levels "$LEVELS" > "$OUT/$name.eval.txt"
  echo "--- $name"; grep -E "level|^all" "$OUT/$name.eval.txt"
done
for scale in 1 2 0.5; do
  f="$OUT/mechanism_x$scale.txt"
  [[ -f "$f" ]] || "${CLI[@]}" fmd-eval --weights "$OUT/cond_s2026.bin" --root data/fmd --levels "$LEVELS" \
    --validation-seed 2026 --sigma-scale "$scale" > "$f"
  echo "--- validation, condition from ${scale}x σ"; grep -E "level|^all" "$f"
done
echo "all Q13 runs done"
