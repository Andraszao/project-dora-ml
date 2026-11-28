#!/usr/bin/env bash
# Brick 32 (quant-fidelity): the fidelity measurements for every model behind docs/RESULTS.md (results recorded there).
#
#   tools/fetch_fmd.sh && tools/fetch_ext.sh, the earlier experiment scripts (for their weights), then
#   tools/experiments/fidelity.sh
#
# FMD test_mix (raw frames, 48 images): the bundled synthetic model, supervised dilated (P3), supervised U-Net (P4),
# the all-level U-Net (Q1), plain Noise2Void dilated (P3) and Noise2Void U-Net (P4). The three external sets (Q18):
# DORA Noise2Void and supervised, CAREamics Noise2Void and CARE. Models whose weights are missing are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/fidelity
mkdir -p "$OUT"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)

measure() {   # measure <name> <fidelity args...>
  local f="$OUT/$1.txt"; shift
  [[ -f "$f" ]] || "${CLI[@]}" fidelity "$@" > "$f"
  echo "--- $(basename "$f" .txt)"; sed -n '4,6p' "$f"
}
for m in godot/models/dora_bio_weights.bin:bundled runs/p3/sup_plain.bin:sup-dilated runs/p4/unet_sup.bin:sup-unet \
         runs/q1/unet_levels60.bin:sup-unet-levels runs/p3/n2v_plain.bin:n2v-dilated runs/p4/unet_n2v.bin:n2v-unet; do
  [[ -f "${m%%:*}" ]] && measure "fmd-${m##*:}" --fmd data/fmd --weights "${m%%:*}"
done
for d in deepbacs-mreb deepbacs-nucleoid zerocost-care2d; do
  measure "$d-dora-n2v" --dataset "$d" --weights "runs/q18/dora_n2v_$d.bin"
  measure "$d-dora-sup" --dataset "$d" --weights "runs/q18/dora_sup_$d.bin"
  measure "$d-careamics-n2v" --dataset "$d" --predictions "runs/q18/careamics_n2v/$d" \
    --label "CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU"
  measure "$d-careamics-care" --dataset "$d" --predictions "runs/q18/careamics_care/$d" \
    --label "CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU"
done
echo "all fidelity measurements done"
