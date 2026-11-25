#!/usr/bin/env bash
# Quality pass Q1 (brick 25): the best supervised recipe (results in docs/RESULTS.md; pre-registered in PATHWAY.md).
#
#   tools/fetch_fmd.sh && tools/experiments/p4_arch.sh && tools/experiments/q1_supervised.sh
#
# Candidate: U-Net (16 -> 64 features, depth 2), supervised, all five FMD noise levels (2 frames each), 60 epochs.
# Ablation: the same at 30 epochs. Baseline: P4's U-Net trained on raw frames only for 30 epochs (runs/p4/unet_sup.bin).
# Everything else as in P3/P4: batch 16, 64 px patches, seed 2026, FOV 19 never used, validation on held-out FOVs.
# Finished runs and evaluations are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/q1 BASE=runs/p4/unet_sup.bin
mkdir -p "$OUT"
[[ -f "$BASE" ]] || { echo "run tools/experiments/p4_arch.sh first"; exit 1; }
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
COMMON=(--data data/fmd --steps 50 --batch 16 --patch 64 --seed 2026 --journey "$OUT/journey.json"
        --mode supervised --arch unet --features 16 --depth 2 --fmd-frames 2 --fmd-levels raw,avg2,avg4,avg8,avg16)
LEVELS=raw,avg2,avg4,avg8,avg16

train() {   # train <name> <epochs> <description>
  [[ -f "$OUT/$1.bin" ]] && return 0
  echo "=== train $1: $3"
  "${CLI[@]}" train "${COMMON[@]}" --epochs "$2" --out "$OUT/$1.bin" --describe "FMD (12 categories, FOV 19 excluded): $3" > "$OUT/$1.train.log"
  tail -2 "$OUT/$1.train.log"
}
train unet_levels60 60 "U-Net, supervised, all 5 noise levels, 60 epochs"
train unet_levels30 30 "U-Net, supervised, all 5 noise levels, 30 epochs"

[[ -f "$OUT/base.levels.txt" ]] || "${CLI[@]}" fmd-eval --weights "$BASE" --root data/fmd --levels avg2,avg4,avg8,avg16 > "$OUT/base.levels.txt"
for name in unet_levels60 unet_levels30; do
  [[ -f "$OUT/$name.eval.txt" ]] || "${CLI[@]}" fmd-eval --weights "$OUT/$name.bin" --root data/fmd --levels "$LEVELS" > "$OUT/$name.eval.txt"
  echo "--- $name"; grep -E "level|^all" "$OUT/$name.eval.txt"
done
echo "--- baseline (raw: runs/p4/unet_sup.eval.txt)"; grep -E "level|^all" "$OUT/base.levels.txt"
echo "all Q1 runs done"
