#!/usr/bin/env bash
# Quality pass Q12 (brick 33): closing the self-supervised gap, pre-registered in PATHWAY.md (commit 1a16b2f) before any
# of its code existed. Results in docs/RESULTS.md.
#
#   tools/fetch_ext.sh && tools/fetch_fmd.sh && tools/experiments/q12_n2v.sh
#
# Three steps, each chosen on validation images (the training images a seed-2026 run holds out), by the pre-registered
# rule: the highest mean validation PSNR over the three sets among candidates whose MreB validation slope is >= 0.85
# (else the highest mean PSNR); ties within 0.05 dB go to the cheaper (earlier-listed) candidate.
#   1. network: dilated (26k) | U-Net 16 (118k) | U-Net 32 (about 0.47M), at 2% masking, 30 x 50 steps
#   2. masking for the chosen network: 2% | 0.2%
#   3. budget: 30 | 90 epochs, only if at least two of step 2's winning runs ended at the 30-epoch cap
# Then the chosen recipe is scored once on the test splits, and trained on FMD's raw frames for the guard.
# Finished runs and evaluations are skipped; the selection log is runs/q12/selection.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/q12
mkdir -p "$OUT"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
DATASETS=(deepbacs-mreb deepbacs-nucleoid zerocost-care2d)
COMMON=(--data data/ext --steps 50 --batch 16 --patch 64 --seed 2026 --mode n2v --journey "$OUT/journey.json")
declare -A ARGS=(
  [dilated]="--arch dilated"
  [unet16]="--arch unet --features 16 --depth 2"
  [unet32]="--arch unet --features 32 --depth 2"
)
LOG="$OUT/selection.txt"

train_candidate() {   # train_candidate <name> <extra args...>: trains on all three sets
  local name="$1" d; shift
  for d in "${DATASETS[@]}"; do
    [[ -f "$OUT/${name}_$d.bin" ]] && continue
    echo "=== $name on $d: $*"
    "${CLI[@]}" train --ext "$d" "${COMMON[@]}" "$@" --out "$OUT/${name}_$d.bin" \
      --describe "$d training split, noisy images only: Q12 candidate $name ($*)" > "$OUT/${name}_$d.train.log"
    tail -1 "$OUT/${name}_$d.train.log"
  done
}

score() {   # score <name>: "<mean validation PSNR> <MreB validation slope>"
  local name="$1" d sum=0 v slope
  for d in "${DATASETS[@]}"; do
    local f="$OUT/val_${name}_$d.txt"
    [[ -f "$f" ]] || "${CLI[@]}" ext-eval --dataset "$d" --weights "$OUT/${name}_$d.bin" --split validation --seed 2026 > "$f"
    v="$(awk '$1 == "mean" { print $8 }' "$f")"; sum="$(awk -v a="$sum" -v b="$v" 'BEGIN { print a + b }')"
  done
  local g="$OUT/valfid_${name}_deepbacs-mreb.txt"
  [[ -f "$g" ]] || "${CLI[@]}" fidelity --dataset deepbacs-mreb --weights "$OUT/${name}_deepbacs-mreb.bin" --split validation --seed 2026 > "$g"
  slope="$(awk '$1 == "method" { print $2 }' "$g")"
  awk -v s="$sum" -v k="$slope" 'BEGIN { printf "%.3f %s\n", s / 3, k }'
}

choose() {   # choose <step> <candidate...>: prints the winner by the pre-registered rule, logs the scores
  local step="$1" c; shift
  for c in "$@"; do echo "$c $(score "$c")"; done | tee -a "$LOG" | awk -v step="$step" '
    { name[NR] = $1; psnr[NR] = $2; slope[NR] = $3; n = NR }
    END {
      best = 0
      for (pass = 1; pass <= 2 && best == 0; pass++)          # pass 1: slope >= 0.85 only; pass 2: everyone
        for (i = 1; i <= n; i++)
          if ((pass == 2 || slope[i] >= 0.85) && (best == 0 || psnr[i] > psnr[best] + 0.05)) best = i
      print name[best]
    }'
}

: > "$LOG"
echo "step 1: network (mean validation PSNR over 3 sets, MreB validation slope)" >> "$LOG"
for c in dilated unet16 unet32; do train_candidate "$c" ${ARGS[$c]}; done
net="$(choose 1 dilated unet16 unet32)"; echo "chosen network: $net" >> "$LOG"

echo "step 2: masking" >> "$LOG"
train_candidate "${net}_m02" ${ARGS[$net]} --mask-ratio 0.002
mask_winner="$(choose 2 "$net" "${net}_m02")"; echo "chosen masking: $([[ "$mask_winner" == "$net" ]] && echo 2% || echo 0.2%)" >> "$LOG"
mask_args=(); [[ "$mask_winner" != "$net" ]] && mask_args=(--mask-ratio 0.002)

capped="$(for d in "${DATASETS[@]}"; do grep -c "stopped: max_epochs" "$OUT/${mask_winner}_$d.train.log" || true; done | awk '{ s += $1 } END { print s }')"
echo "step 3: budget ($capped of 3 winning runs ended at the 30-epoch cap)" >> "$LOG"
final="$mask_winner"
if [[ "$capped" -ge 2 ]]; then
  train_candidate "${mask_winner}_e90" ${ARGS[$net]} "${mask_args[@]}" --epochs 90
  final="$(choose 3 "$mask_winner" "${mask_winner}_e90")"
else
  echo "budget step skipped (fewer than two runs at the cap)" >> "$LOG"
fi
echo "final recipe: $final" >> "$LOG"
final_args=(${ARGS[$net]} "${mask_args[@]}"); if [[ "$final" == *_e90 ]]; then final_args+=(--epochs 90); else final_args+=(--epochs 30); fi
echo "final arguments: ${final_args[*]}" >> "$LOG"

for d in "${DATASETS[@]}"; do
  [[ -f "$OUT/test_$d.txt" ]] || "${CLI[@]}" ext-eval --dataset "$d" --weights "$OUT/${final}_$d.bin" > "$OUT/test_$d.txt"
  [[ -f "$OUT/testfid_$d.txt" ]] || "${CLI[@]}" fidelity --dataset "$d" --weights "$OUT/${final}_$d.bin" > "$OUT/testfid_$d.txt"
done
if [[ ! -f "$OUT/fmd_guard.bin" ]]; then
  echo "=== FMD guard: ${final_args[*]}"
  "${CLI[@]}" train --data data/fmd --steps 50 --batch 16 --patch 64 --seed 2026 --fmd-frames 4 --fmd-levels raw \
    --mode n2v --struct off "${final_args[@]}" --journey "$OUT/journey.json" --out "$OUT/fmd_guard.bin" \
    --describe "FMD (12 categories, FOV 19 excluded): Q12's chosen Noise2Void recipe, raw frames, plain blind spots" > "$OUT/fmd_guard.train.log"
  tail -1 "$OUT/fmd_guard.train.log"
fi
[[ -f "$OUT/fmd_guard.eval.txt" ]] || "${CLI[@]}" fmd-eval --weights "$OUT/fmd_guard.bin" --root data/fmd --levels raw > "$OUT/fmd_guard.eval.txt"
cat "$LOG"
for d in "${DATASETS[@]}"; do echo "--- $d test: $(grep '^mean' "$OUT/test_$d.txt" | cut -c14-) | fidelity $(grep '^method' "$OUT/testfid_$d.txt" | cut -c11-)"; done
echo "--- FMD guard: $(grep '^all' "$OUT/fmd_guard.eval.txt")"
echo "all Q12 runs done"
