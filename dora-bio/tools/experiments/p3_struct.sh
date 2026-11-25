#!/usr/bin/env bash
# P3 brick 15: structured blind spots on real FMD data (results in docs/RESULTS.md).
#
#   tools/fetch_fmd.sh && tools/experiments/p3_struct.sh
#
# Same budget and seed as P2 (30 epochs × 50 steps × batch 16 × 64 px, seed 2026, 4 raw frames per FOV, FOV 19 never
# used), but with P3's validation set, which covers every held-out FOV; so the plain baselines are retrained too.
# Structured blind spots go only on the training images the noise check flags (--struct auto). The footprint is
# chosen on validation PSNR, never on test_mix. Every candidate is still scored on test_mix and reported, but the
# chosen one is what gate 15 judges. Runs and evaluations already finished (their files in runs/p3/) are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/p3
mkdir -p "$OUT"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
COMMON=(--data data/fmd --epochs 30 --steps 50 --batch 16 --patch 64 --seed 2026 --fmd-frames 4 --fmd-levels raw
        --journey "$OUT/journey.json")
FOOTPRINTS=(square-1 diamond-2 square-2)

train() {   # train <name> <description> <train args...>
  local name="$1" description="$2"; shift 2
  [[ -f "$OUT/$name.bin" ]] && return 0
  echo "=== train $name: $description"
  "${CLI[@]}" train "${COMMON[@]}" --out "$OUT/$name.bin" --describe "FMD (12 categories, FOV 19 excluded): $description" "$@" \
    > "$OUT/$name.train.log"
  tail -2 "$OUT/$name.train.log"
}
val_psnr() { sed -n 's/.*val PSNR \([0-9.]*\) dB vs noisy.*/\1/p' "$OUT/$1.train.log" | tail -1; }

train n2v_plain "Noise2Void, raw frames, plain blind spots" --mode n2v --struct off
train sup_plain "supervised raw -> ground truth" --mode supervised
for f in "${FOOTPRINTS[@]}"; do
  train "n2v_struct_$f" "Noise2Void, raw frames, structured blind spots ($f) on flagged images" --mode n2v --struct auto --footprint "$f"
done

# The footprint is chosen on validation PSNR (held-out training FOVs), never on test_mix.
best="" best_psnr=0
for f in "${FOOTPRINTS[@]}"; do
  p="$(val_psnr "n2v_struct_$f")"
  echo "validation PSNR with $f: $p dB"
  if awk -v a="$p" -v b="$best_psnr" 'BEGIN { exit !(a > b) }'; then best="$f" best_psnr="$p"; fi
done
echo "plain Noise2Void validation PSNR: $(val_psnr n2v_plain) dB; chosen footprint: $best"
echo "$best" > "$OUT/chosen_footprint.txt"
cp "$OUT/n2v_struct_$best.bin" "$OUT/n2v_struct.bin"

# Second design, after the first missed the bar on confocal (bar unchanged; see PATHWAY.md): two networks routed by
# the noise check, plain on the images it calls independent and structured on the ones it calls correlated.
train n2v_indep "Noise2Void, raw frames, images whose noise check says independent, plain blind spots" \
  --mode n2v --struct off --subset independent
train n2v_corr "Noise2Void, raw frames, images whose noise check says correlated, structured blind spots ($best)" \
  --mode n2v --struct auto --footprint "$best" --subset correlated

evaluate() {   # evaluate <name> <fmd-eval model args...>; skips evaluations already done
  local name="$1"; shift
  [[ -f "$OUT/$name.eval.txt" ]] || "${CLI[@]}" fmd-eval "$@" --root data/fmd --levels raw > "$OUT/$name.eval.txt"
  [[ -f "$OUT/$name.levels.txt" ]] || "${CLI[@]}" fmd-eval "$@" --root data/fmd --levels avg2,avg4,avg8,avg16 > "$OUT/$name.levels.txt"
  echo "--- $name"; cat "$OUT/$name.eval.txt"
}
for name in n2v_plain n2v_struct sup_plain; do evaluate "$name" --weights "$OUT/$name.bin"; done
evaluate n2v_routed --weights "$OUT/n2v_indep.bin" --weights-correlated "$OUT/n2v_corr.bin"
for f in "${FOOTPRINTS[@]}"; do   # reported for completeness; not used to choose
  [[ -f "$OUT/n2v_struct_$f.eval.txt" ]] \
    || "${CLI[@]}" fmd-eval --weights "$OUT/n2v_struct_$f.bin" --root data/fmd --levels raw > "$OUT/n2v_struct_$f.eval.txt"
  echo "--- n2v_struct_$f (not used for the choice)"; cat "$OUT/n2v_struct_$f.eval.txt"
done
echo "all P3 structured-mask runs done"
