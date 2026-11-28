#!/usr/bin/env bash
# Quality pass Q10 (brick 27): route each image by its noise level (results in docs/RESULTS.md; pre-registered in PATHWAY.md).
#
#   tools/fetch_fmd.sh && tools/experiments/p4_arch.sh && tools/experiments/q1_supervised.sh && tools/experiments/q10_routing.sh
#
# A = P4's supervised U-Net trained on raw frames, B = Q1's all-level U-Net; no new training. The threshold is chosen on
# the 46 validation FOVs both runs held out (seed 2026), then test_mix is scored once with it.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/q10 A=runs/p4/unet_sup.bin B=runs/q1/unet_levels60.bin
mkdir -p "$OUT"
[[ -f "$A" && -f "$B" ]] || { echo "run tools/experiments/p4_arch.sh and tools/experiments/q1_supervised.sh first"; exit 1; }
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)

[[ -f "$OUT/route.txt" ]] || "${CLI[@]}" fmd-route --weights "$A" --weights-clean "$B" --root data/fmd --seed 2026 > "$OUT/route.txt"
cat "$OUT/route.txt"
threshold="$(sed -n 's/^chosen noise threshold: //p' "$OUT/route.txt")"
[[ -f "$OUT/routed.eval.txt" ]] || "${CLI[@]}" fmd-eval --weights "$A" --weights-clean "$B" --noise-threshold "$threshold" \
  --root data/fmd --levels raw,avg2,avg4,avg8,avg16 > "$OUT/routed.eval.txt"
echo "--- routed on test_mix"; grep -E "level|^all|routed" "$OUT/routed.eval.txt"
echo "all Q10 runs done"
