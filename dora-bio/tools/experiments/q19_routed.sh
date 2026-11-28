#!/usr/bin/env bash
# Quality pass Q19 (brick 37): the U-Net with routing, pre-registered in PATHWAY.md (commit 02496b1) before any of its
# code existed. Results in docs/RESULTS.md.
#
#   tools/fetch_ext.sh && tools/fetch_fmd.sh && tools/experiments/p3_struct.sh && tools/experiments/q19_routed.sh
#
# 1. Guard 1: the noise check's threshold, kept or re-chosen on the validation FOVs of a seed-2026 run (never test).
# 2. FMD: the U-Net under Q2's rule C on the images the noise check calls independent; for the ones it calls
#    correlated, P3's dilated diamond-2 network or a U-Net with diamond-2 blind spots under rule C, chosen by widefield
#    validation PSNR (within 0.05 dB: P3's). Then the routed pair is scored once on test_mix raw (unrounded).
# 3. Public sets: the same recipe per set (the noise check routes training and test images), CARE's protocol.
# 4. The verdict, by the pre-registered rule. Finished runs and evaluations are skipped; the log is runs/q19/selection.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/q19 P3=runs/p3
mkdir -p "$OUT"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
DATASETS=(deepbacs-mreb deepbacs-nucleoid zerocost-care2d)
UNET_C=(--arch unet --features 16 --depth 2 --select loss --patience 0 --keep last --epochs 30)
DILATED_P3=(--arch dilated --epochs 30)                       # P3's correlated network: dilated, today's rule
FMD_COMMON=(--data data/fmd --steps 50 --batch 16 --patch 64 --seed 2026 --fmd-frames 4 --fmd-levels raw --mode n2v
            --journey "$OUT/journey.json")
EXT_COMMON=(--data data/ext --steps 50 --batch 16 --patch 64 --seed 2026 --mode n2v --journey "$OUT/journey.json")
STRUCT=(--struct auto --footprint diamond-2)
LOG="$OUT/selection.txt"
[[ -f "$P3/n2v_corr.bin" ]] || { echo "needs P3's correlated network (tools/experiments/p3_struct.sh)"; exit 1; }

# 1. Guard 1.
[[ -f "$OUT/threshold.txt" ]] || "${CLI[@]}" fmd-diagnose --root data/fmd --levels raw --validation-seed 2026 --choose-threshold > "$OUT/threshold.txt"
threshold="$(sed -n 's/^routing threshold: \([0-9.]*\).*/\1/p' "$OUT/threshold.txt")"
{ echo "guard 1: $(grep '^routing threshold' "$OUT/threshold.txt")"; } > "$LOG"
[[ "$threshold" == "0.400" ]] || { echo "the threshold moved on validation; routing at $threshold needs --noise-threshold plumbing first" | tee -a "$LOG"; exit 1; }

train_fmd() {   # train_fmd <name> <args...>
  local name="$1"; shift
  [[ -f "$OUT/$name.bin" ]] && return 0
  echo "=== FMD $name: $*"
  "${CLI[@]}" train "${FMD_COMMON[@]}" "$@" --out "$OUT/$name.bin" > "$OUT/$name.train.log"
  tail -1 "$OUT/$name.train.log"
}

# 2. FMD.
train_fmd fmd_unet_indep --subset independent --struct off "${UNET_C[@]}"
train_fmd fmd_unet_corr --subset correlated "${STRUCT[@]}" "${UNET_C[@]}"
for c in p3 unet; do
  corr="$P3/n2v_corr.bin"; [[ "$c" == unet ]] && corr="$OUT/fmd_unet_corr.bin"
  [[ -f "$OUT/val_fmd_$c.txt" ]] || "${CLI[@]}" fmd-eval --weights "$OUT/fmd_unet_indep.bin" --weights-correlated "$corr" \
    --root data/fmd --levels raw --validation-seed 2026 > "$OUT/val_fmd_$c.txt"
done
{
  echo "correlated network, widefield validation PSNR (routed / widefield images):"
  for c in p3 unet; do
    echo "$c $(awk '$1 == "WideField" { print $9 }' "$OUT/val_fmd_$c.txt") $(sed -n 's/^routed by the noise check: \([0-9]*\) of.*/\1/p' "$OUT/val_fmd_$c.txt")/$(awk '$1 == "WideField" { print $2 }' "$OUT/val_fmd_$c.txt")"
  done
} >> "$LOG"
choice="$(awk '$1 == "p3" { p = $2 } $1 == "unet" { u = $2 } END { print (u > p + 0.05 + 1e-9) ? "unet" : "p3" }' "$LOG")"
echo "chosen correlated network: $choice" >> "$LOG"
corr_fmd="$P3/n2v_corr.bin"; [[ "$choice" == unet ]] && corr_fmd="$OUT/fmd_unet_corr.bin"
[[ -f "$OUT/test_fmd.txt" ]] || "${CLI[@]}" fmd-eval --weights "$OUT/fmd_unet_indep.bin" --weights-correlated "$corr_fmd" \
  --root data/fmd --levels raw --precise > "$OUT/test_fmd.txt"

# 3. Public sets: per set, an independent-image U-Net and/or a correlated-image network, whichever images exist.
CORR_ARGS=("${STRUCT[@]}" "${DILATED_P3[@]}"); [[ "$choice" == unet ]] && CORR_ARGS=("${STRUCT[@]}" "${UNET_C[@]}")
train_ext() {   # train_ext <set> <kind> <args...>: prints nothing when the set has no image of that kind
  local d="$1" kind="$2"; shift 2
  local out="$OUT/${d}_$kind.bin" log="$OUT/${d}_$kind.train.log"
  [[ -f "$out" || -f "$OUT/${d}_$kind.none" ]] && return 0
  echo "=== $d $kind: $*"
  if ! "${CLI[@]}" train --ext "$d" "${EXT_COMMON[@]}" --subset "$kind" "$@" --out "$out" \
       --describe "$d training split, noisy images only, those the noise check calls $kind: Q19" > "$log" 2>&1; then
    grep -q "No training image's noise check says $kind" "$log" && { touch "$OUT/${d}_$kind.none"; echo "  (no $kind images)"; return 0; }
    cat "$log"; return 1
  fi
  tail -1 "$log"
}
routed_args() {   # routed_args <set>: the ext-eval/fidelity arguments for the set's routed pair
  local d="$1" args=()
  if [[ -f "$OUT/${d}_independent.bin" ]]; then
    args=(--weights "$OUT/${d}_independent.bin")
    [[ -f "$OUT/${d}_correlated.bin" ]] && args+=(--weights-correlated "$OUT/${d}_correlated.bin")
  else
    args=(--weights "$OUT/${d}_correlated.bin")
  fi
  echo "${args[@]}"
}
for d in "${DATASETS[@]}"; do
  train_ext "$d" independent --struct off "${UNET_C[@]}"
  train_ext "$d" correlated "${CORR_ARGS[@]}"
  # shellcheck disable=SC2046
  [[ -f "$OUT/test_$d.txt" ]] || "${CLI[@]}" ext-eval --dataset "$d" $(routed_args "$d") > "$OUT/test_$d.txt"
done
# shellcheck disable=SC2046
[[ -f "$OUT/testfid_deepbacs-mreb.txt" ]] || "${CLI[@]}" fidelity --dataset deepbacs-mreb $(routed_args deepbacs-mreb) > "$OUT/testfid_deepbacs-mreb.txt"

# 4. The verdict.
{
  for d in "${DATASETS[@]}"; do echo "set $d $(awk '$1 == "mean" { print $8 }' "$OUT/test_$d.txt")"; done
  echo "slope $(awk '$1 == "method" { print $2 }' "$OUT/testfid_deepbacs-mreb.txt")"
  for m in Confocal TwoPhoton WideField; do echo "fmd $m $(awk -v m="$m" '$1 == m { print $9 }' "$OUT/test_fmd.txt")"; done
  echo "fmd all $(awk '$1 == "all," { print $10 }' "$OUT/test_fmd.txt")"
} > "$OUT/test_summary.txt"
awk '
  $1 == "set" { s[$2] = $3; mean += $3 / 3 } $1 == "slope" { slope = $2 } $1 == "fmd" { f[$2] = $3 }
  END {
    ok = f["all"] >= 33.17 - 1e-9 && f["Confocal"] >= 33.99 - 1e-9 && f["TwoPhoton"] >= 32.36 - 1e-9 && f["WideField"] >= 31.66 - 1e-9
    ok = ok && mean >= 26.70 - 1e-9 && slope >= 0.90 - 1e-9
    ok = ok && s["deepbacs-mreb"] >= 25.73 - 1e-9 && s["deepbacs-nucleoid"] >= 33.61 - 1e-9 && s["zerocost-care2d"] >= 19.20 - 1e-9
    printf "test: FMD %.4f (confocal %.2f, two-photon %.2f, widefield %.2f); public mean %.3f (MreB %.2f, nucleoid %.2f, CARE 2D %.2f), MreB slope %.2f\n", \
      f["all"], f["Confocal"], f["TwoPhoton"], f["WideField"], mean, s["deepbacs-mreb"], s["deepbacs-nucleoid"], s["zerocost-care2d"], slope
    printf "verdict: %s\n", ok ? "adopted" : "not-adopted"
  }' "$OUT/test_summary.txt" >> "$LOG"
cat "$LOG"
