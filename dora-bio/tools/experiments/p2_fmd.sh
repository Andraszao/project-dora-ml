#!/usr/bin/env bash
# P2 experiments: DORA on the real FMD dataset (results in docs/RESULTS.md).
#
#   tools/fetch_fmd.sh && tools/experiments/p2_fmd.sh
#
# Every run uses the same seed and the same number of steps (30 epochs × 50 steps × batch 16 × 64 px), and FMD's
# test FOV (19) is always excluded from training. Training is deterministic, so rerunning reproduces the same
# weights and numbers. Runs already finished (runs/p2/<name>.bin present) are not retrained. Weights stay in the
# gitignored runs/ folder: they are trained on CC BY-SA data, so publishing them is a separate decision.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/p2
mkdir -p "$OUT"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
COMMON=(--data data/fmd --epochs 30 --steps 50 --batch 16 --patch 64 --seed 2026 --journey "$OUT/journey.json")

run() {   # run <name> <description> <train args...>
  local name="$1" description="$2"; shift 2
  if [[ ! -f "$OUT/$name.bin" ]]; then
    echo "=== train $name: $description"
    "${CLI[@]}" train "${COMMON[@]}" --out "$OUT/$name.bin" --describe "FMD (12 categories, FOV 19 excluded): $description" "$@" \
      > "$OUT/$name.train.log"
    tail -2 "$OUT/$name.train.log"
  fi
  "${CLI[@]}" fmd-eval --weights "$OUT/$name.bin" --root data/fmd --levels raw > "$OUT/$name.eval.txt"
  echo "--- $name"; cat "$OUT/$name.eval.txt"
}

# Baseline: the bundled model, trained only on synthetic data.
"${CLI[@]}" fmd-eval --weights godot/models/dora_bio_weights.bin --root data/fmd --levels raw > "$OUT/synthetic.eval.txt"
echo "--- synthetic"; cat "$OUT/synthetic.eval.txt"

run n2v_raw    "Noise2Void, raw frames only, 4 per FOV"                   --mode n2v --route off --fmd-frames 4 --fmd-levels raw
run sup_raw    "supervised raw -> ground truth, 4 frames per FOV"         --mode supervised --fmd-frames 4 --fmd-levels raw
# v1 idea 1: v1's geometric augmentation (±15° rotation, 0.85-1.15 scale, nearest neighbour) instead of lossless flips.
run n2v_v1aug  "Noise2Void, raw frames, v1 rotate+scale augmentation"     --mode n2v --route off --fmd-frames 4 --fmd-levels raw --augment v1
run sup_v1aug  "supervised, raw frames, v1 rotate+scale augmentation"     --mode supervised --fmd-frames 4 --fmd-levels raw --augment v1
# v1 idea 2: graded noise. FMD's 2/4/8/16-frame averages are real versions of v1's "progressive noise" training.
run n2v_levels "Noise2Void, all 5 noise levels, 2 frames each per FOV"    --mode n2v --route off --fmd-frames 2 --fmd-levels raw,avg2,avg4,avg8,avg16
run sup_levels "supervised, all 5 noise levels, 2 frames each per FOV"    --mode supervised --fmd-frames 2 --fmd-levels raw,avg2,avg4,avg8,avg16

# Graded noise is meant to make a model that also works on cleaner inputs, so score the raw-only and all-level
# models at FMD's four lower noise levels too.
for name in synthetic n2v_raw sup_raw n2v_levels sup_levels; do
  weights="$OUT/$name.bin"; [[ "$name" == synthetic ]] && weights=godot/models/dora_bio_weights.bin
  "${CLI[@]}" fmd-eval --weights "$weights" --root data/fmd --levels avg2,avg4,avg8,avg16 > "$OUT/$name.levels.txt"
  echo "--- $name at the lower noise levels"; cat "$OUT/$name.levels.txt"
done

# Noise2Void assumes noise is independent between neighbouring pixels. Measure it on the real test images.
"${CLI[@]}" fmd-noise --root data/fmd --levels raw > "$OUT/noise.txt"
echo "--- noise correlation"; cat "$OUT/noise.txt"
echo "all P2 runs done"
