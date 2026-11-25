#!/usr/bin/env bash
# P3 exit: Noise2Void with and without variance stabilization on real FMD data (results in docs/RESULTS.md).
#
#   tools/fetch_fmd.sh && tools/experiments/p3_struct.sh && tools/experiments/p3_vst.sh
#
# Same budget, seed and validation as p3_struct.sh. Each training image is calibrated on its own (NoiseCalibration),
# transformed (generalized Anscombe), and the network learns in that domain; test images get the same treatment and
# the unbiased inverse. Compared with the matching designs without it: plain (n2v_plain) and routed (n2v_routed).
# "Wins" was set before running (PATHWAY.md, P3 exit): all-48 PSNR up at raw, and down by no more than 0.2 dB at any
# lower noise level. Runs and evaluations already finished are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/p3
[[ -f "$OUT/n2v_plain.eval.txt" && -f "$OUT/chosen_footprint.txt" ]] || { echo "run tools/experiments/p3_struct.sh first"; exit 1; }
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
COMMON=(--data data/fmd --epochs 30 --steps 50 --batch 16 --patch 64 --seed 2026 --fmd-frames 4 --fmd-levels raw
        --journey "$OUT/journey.json" --mode n2v --vst)
footprint="$(cat "$OUT/chosen_footprint.txt")"

train() {   # train <name> <description> <train args...>
  local name="$1" description="$2"; shift 2
  [[ -f "$OUT/$name.bin" ]] && return 0
  echo "=== train $name: $description"
  "${CLI[@]}" train "${COMMON[@]}" --out "$OUT/$name.bin" --describe "FMD (12 categories, FOV 19 excluded): $description" "$@" \
    > "$OUT/$name.train.log"
  tail -2 "$OUT/$name.train.log"
}
evaluate() {   # evaluate <name> <fmd-eval model args...>
  local name="$1"; shift
  [[ -f "$OUT/$name.eval.txt" ]] || "${CLI[@]}" fmd-eval "$@" --vst --root data/fmd --levels raw > "$OUT/$name.eval.txt"
  [[ -f "$OUT/$name.levels.txt" ]] || "${CLI[@]}" fmd-eval "$@" --vst --root data/fmd --levels avg2,avg4,avg8,avg16 > "$OUT/$name.levels.txt"
  echo "--- $name"; cat "$OUT/$name.eval.txt"
}

train n2v_plain_vst "Noise2Void, raw frames, plain blind spots, variance-stabilized" --struct off
train n2v_indep_vst "Noise2Void, raw frames, independent-noise images, plain blind spots, variance-stabilized" \
  --struct off --subset independent
train n2v_corr_vst "Noise2Void, raw frames, correlated-noise images, structured blind spots ($footprint), variance-stabilized" \
  --struct auto --footprint "$footprint" --subset correlated

evaluate n2v_plain_vst --weights "$OUT/n2v_plain_vst.bin"
evaluate n2v_routed_vst --weights "$OUT/n2v_indep_vst.bin" --weights-correlated "$OUT/n2v_corr_vst.bin"
echo "all P3 variance-stabilization runs done"
