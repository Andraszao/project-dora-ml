#!/usr/bin/env bash
# Quality pass Q2 (brick 35): when a Noise2Void U-Net should stop, pre-registered in PATHWAY.md (commit 74153da) before
# any of its code existed. Results in docs/RESULTS.md.
#
#   tools/fetch_ext.sh && tools/fetch_fmd.sh && tools/experiments/q12_n2v.sh && tools/experiments/q2_stop.sh
#
# Candidates for Q12's network (U-Net 16, depth 2, 2% masking, at most 30 x 50 steps):
#   A  today's rule: choose and stop (patience 6) on validation PSNR when clean images exist, else on the masked loss
#      (Q12's runs: runs/q12/unet16_<set>.bin and runs/q12/fmd_guard.bin)
#   B  --select loss --patience 0: the masked validation loss always, all 30 epochs, best epoch kept
#   C  B's run, last epoch kept (--save-last)
# Chosen by mean validation PSNR over four sets (three public sets' seed-2026 validation images, CARE protocol; FMD's
# seed-2026 validation FOVs, raw). Ties within 0.05 dB go to C, then B. If A wins, the pass stops: not adopted.
# Otherwise the winner is scored once on test (public test splits, MreB slope, FMD test_mix raw), and the dilated
# network is trained and tested under the same rule for the rule-only branch. The verdict follows the pre-registered
# rule; finished runs and evaluations are skipped; the log is runs/q2/selection.txt.
# FMD test scores are the unrounded mean (`fmd-eval --precise`). That was added after the first run: the chosen U-Net
# printed 31.48 against a 31.48 guard, and only the unrounded value (31.4794) can decide it. Q12 compared its bar the
# same way (26.697 against 26.70).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/q2 Q12=runs/q12
mkdir -p "$OUT"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
DATASETS=(deepbacs-mreb deepbacs-nucleoid zerocost-care2d)
EXT_COMMON=(--data data/ext --steps 50 --batch 16 --patch 64 --seed 2026 --mode n2v --journey "$OUT/journey.json")
FMD_COMMON=(--data data/fmd --steps 50 --batch 16 --patch 64 --seed 2026 --fmd-frames 4 --fmd-levels raw --mode n2v
            --struct off --journey "$OUT/journey.json")
RULE=(--select loss --patience 0 --epochs 30)
declare -A NET=([unet]="--arch unet --features 16 --depth 2" [dilated]="--arch dilated")
LOG="$OUT/selection.txt"
: > "$LOG"

for f in "${DATASETS[@]/#/$Q12/unet16_}" "$Q12/fmd_guard"; do
  [[ -f "$f.bin" ]] || { echo "needs Q12's run $f.bin (tools/experiments/q12_n2v.sh)"; exit 1; }
done

train_rule() {   # train_rule <net>: B and C weights for every set, from one run each
  local net="$1" d
  for d in "${DATASETS[@]}"; do
    [[ -f "$OUT/${net}_B_$d.bin" && -f "$OUT/${net}_C_$d.bin" ]] && continue
    echo "=== $net on $d: ${NET[$net]} ${RULE[*]}"
    "${CLI[@]}" train --ext "$d" "${EXT_COMMON[@]}" ${NET[$net]} "${RULE[@]}" \
      --out "$OUT/${net}_B_$d.bin" --save-last "$OUT/${net}_C_$d.bin" \
      --describe "$d training split, noisy images only: Q2 $net, masked-loss selection, full budget" > "$OUT/${net}_$d.train.log"
    tail -2 "$OUT/${net}_$d.train.log"
  done
  if [[ ! -f "$OUT/${net}_B_fmd.bin" || ! -f "$OUT/${net}_C_fmd.bin" ]]; then
    echo "=== $net on FMD: ${NET[$net]} ${RULE[*]}"
    "${CLI[@]}" train "${FMD_COMMON[@]}" ${NET[$net]} "${RULE[@]}" \
      --out "$OUT/${net}_B_fmd.bin" --save-last "$OUT/${net}_C_fmd.bin" > "$OUT/${net}_fmd.train.log"
    tail -2 "$OUT/${net}_fmd.train.log"
  fi
}

weights() {   # weights <net> <candidate> <set|fmd>: the weights file a candidate uses
  local net="$1" c="$2" d="$3"
  if [[ "$c" == A ]]; then [[ "$d" == fmd ]] && echo "$Q12/fmd_guard.bin" || echo "$Q12/unet16_$d.bin"
  else echo "$OUT/${net}_${c}_$d.bin"; fi
}

validation() {   # validation <candidate>: "<mean of 4> <mreb> <nucleoid> <care2d> <fmd>" validation PSNRs (U-Net)
  local c="$1" d f v vals=()
  for d in "${DATASETS[@]}"; do
    f="$OUT/val_unet_${c}_$d.txt"
    [[ -f "$f" ]] || "${CLI[@]}" ext-eval --dataset "$d" --weights "$(weights unet "$c" "$d")" --split validation --seed 2026 > "$f"
    vals+=("$(awk '$1 == "mean" { print $8 }' "$f")")
  done
  f="$OUT/val_unet_${c}_fmd.txt"
  [[ -f "$f" ]] || "${CLI[@]}" fmd-eval --weights "$(weights unet "$c" fmd)" --root data/fmd --levels raw --validation-seed 2026 > "$f"
  vals+=("$(awk '$1 == "all" { print $9 }' "$f")")
  awk -v a="${vals[0]}" -v b="${vals[1]}" -v c="${vals[2]}" -v d="${vals[3]}" \
    'BEGIN { printf "%.3f %s %s %s %s\n", (a + b + c + d) / 4, a, b, c, d }'
}

train_rule unet
echo "validation (U-Net): mean of 4, then MreB, nucleoid, CARE 2D, FMD" >> "$LOG"
for c in A B C; do echo "$c $(validation "$c")"; done | tee -a "$LOG" > "$OUT/val_scores.txt"
winner="$(awk '
  { mean[$1] = $2; if ($2 > best) best = $2 }
  END { n = split("C B A", order, " ")                      # preference within 0.05 dB: C, then B, then A
        for (i = 1; i <= n; i++) if (mean[order[i]] >= best - 0.05 - 1e-9) { print order[i]; exit } }' "$OUT/val_scores.txt")"
echo "chosen: $winner" >> "$LOG"
if [[ "$winner" == A ]]; then
  echo "verdict: not-adopted (today's rule wins on validation; test sets not touched)" >> "$LOG"
  cat "$LOG"; exit 0
fi

test_net() {   # test_net <net>: test PSNRs (public sets, CARE protocol) + MreB slope + FMD test_mix raw, winner's weights
  local net="$1" d
  for d in "${DATASETS[@]}"; do
    [[ -f "$OUT/test_${net}_$d.txt" ]] || "${CLI[@]}" ext-eval --dataset "$d" --weights "$(weights "$net" "$winner" "$d")" > "$OUT/test_${net}_$d.txt"
  done
  [[ -f "$OUT/testfid_${net}_deepbacs-mreb.txt" ]] || "${CLI[@]}" fidelity --dataset deepbacs-mreb \
    --weights "$(weights "$net" "$winner" deepbacs-mreb)" > "$OUT/testfid_${net}_deepbacs-mreb.txt"
  [[ -f "$OUT/test_${net}_fmd.txt" ]] || "${CLI[@]}" fmd-eval --weights "$(weights "$net" "$winner" fmd)" --root data/fmd --levels raw --precise > "$OUT/test_${net}_fmd.txt"
  local m n c fmd slope
  m="$(awk '$1 == "mean" { print $8 }' "$OUT/test_${net}_deepbacs-mreb.txt")"
  n="$(awk '$1 == "mean" { print $8 }' "$OUT/test_${net}_deepbacs-nucleoid.txt")"
  c="$(awk '$1 == "mean" { print $8 }' "$OUT/test_${net}_zerocost-care2d.txt")"
  fmd="$(awk '$1 == "all," { print $10 }' "$OUT/test_${net}_fmd.txt")"   # the unrounded line
  slope="$(awk '$1 == "method" { print $2 }' "$OUT/testfid_${net}_deepbacs-mreb.txt")"
  awk -v n="$net" -v a="$m" -v b="$n" -v c="$c" -v f="$fmd" -v s="$slope" \
    'BEGIN { printf "test %s: public mean %.3f (MreB %s, nucleoid %s, CARE 2D %s), MreB slope %s, FMD raw %s\n", n, (a + b + c) / 3, a, b, c, s, f }'
}

test_net unet | tee -a "$LOG"
train_rule dilated
test_net dilated | tee -a "$LOG"

# The pre-registered adoption rule.
awk '
  $1 == "test" { net = $2; sub(":", "", net); mean[net] = $5; slope[net] = $15 + 0; fmd[net] = $NF }
  END {
    v = "not-adopted"
    if (mean["unet"] >= 26.70 - 1e-9 && slope["unet"] >= 0.90 - 1e-9 && fmd["unet"] >= 31.48 - 1e-9) v = "adopted"
    else if (mean["unet"] >= 26.58 - 1e-9 && slope["unet"] >= 0.75 - 1e-9 && fmd["unet"] >= 31.48 - 1e-9) v = "improved"
    else if (mean["dilated"] >= 26.18 - 1e-9 && fmd["dilated"] >= 31.58 - 1e-9) v = "rule-only"
    printf "verdict: %s\n", v
  }' "$LOG" >> "$LOG"
cat "$LOG"
