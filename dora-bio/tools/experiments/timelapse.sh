#!/usr/bin/env bash
# Brick 40 (Track A): the time-lapse modes on real data, pre-registered in PATHWAY.md (commit da40edd) before any of its
# code existed. Four arms on the same U-Net (16 base features, depth 2) under Q2's rule C, one matched budget
# (60 x 50 steps, batch 16, 64 px patches, seed 2026, CPU), on DeepCAD-RT's zebrafish optic tectum (mostly still) and
# neutrophil pair 02 (motile), scored frame by frame against their aligned high-SNR movies. Results in docs/RESULTS.md.
#
#   tools/fetch_ext.sh deepcad-zebrafish-ot deepcad-neutrophils-02
#   tools/experiments/timelapse.sh
#
# Blocks (every one a standalone sequence, reflected at its own ends): training 0-1199, validation 1210-1359, test
# 1370-1669 (the only block the adoption rule reads) and, on the neutrophils only, a late test block 5000-5299 (reported,
# never gating). Finished steps are skipped. Writes runs/timelapse/summary.txt and the reading
# (runs/timelapse/reading.txt, tools/experiments/timelapse_reading.awk).
#
#   tools/experiments/timelapse.sh --rescore DIR
# scores every finished run again into DIR (DIR/summary.txt, DIR/reading.txt) and trains nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/timelapse EVAL=runs/timelapse RESCORE=0
if [[ "${1:-}" == --rescore ]]; then EVAL="${2:?--rescore needs a folder}" RESCORE=1; fi
mkdir -p "$OUT" "$EVAL"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
# DORA_CLI: a frozen copy of the built CLI, so rebuilding the tree during a long run can't change later steps.
CLI=(dotnet "${DORA_CLI:-tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll}")

declare -A LOW HIGH
LOW[zebrafish]=data/ext/deepcad-zebrafish-ot/01_ZebrafishOT_GCaMP6s_492x492x6955_lowSNR.tif
HIGH[zebrafish]=data/ext/deepcad-zebrafish-ot/01_ZebrafishOT_GCaMP6s_492x492x6955_highSNR.tif
LOW[neutrophils]=data/ext/deepcad-neutrophils-02/02_neutrophil_0.465umPerPixel_512x512x5706_lowSNR.tif
HIGH[neutrophils]=data/ext/deepcad-neutrophils-02/02_neutrophil_0.465umPerPixel_512x512x5706_highSNR.tif
MOVIES=(zebrafish neutrophils)
ARMS=(A1 A2 A3 A4)
declare -A MODE=(
  [A1]="--mode n2v --route off"            # single-frame Noise2Void (today's default network and rule on these movies)
  [A2]="--mode n2n"                        # one-sided Noise2Noise: t -> t+1 and t+1 -> t
  [A3]="--mode n2v --frames 3 --route off" # the 3-frame spatiotemporal blind spot
  [A4]="--mode skip-centre"                # skip-centre Noise2Noise: input t-1, t+1, target t
)
RULE_C=(--arch unet --features 16 --depth 2 --select loss --patience 0 --keep last)
BLOCKS=(--frame-range 0-1199 --validation-range 1210-1359)
BUDGET=(--steps 50 --batch 16 --patch 64 --seed 2026)
TEST=1370-1669 LATE=5000-5299

for m in "${MOVIES[@]}"; do
  [[ -f "${LOW[$m]}" && -f "${HIGH[$m]}" ]] || { echo "missing ${LOW[$m]} or its reference: tools/fetch_ext.sh" >&2; exit 1; }
done

# 1. Training: 4 arms x 2 movies, 60 epochs each.
train() {   # train <movie> <arm> <epochs> <name>
  local m="$1" a="$2" epochs="$3" name="$4"
  [[ -f "$OUT/$name.bin" ]] && return 0
  (( RESCORE )) && { echo "--rescore: $OUT/$name.bin is missing; run tools/experiments/timelapse.sh first" >&2; exit 1; }
  echo "=== $name: ${MODE[$a]}, $epochs x 50 steps"
  # shellcheck disable=SC2086  # MODE holds several options on purpose
  "${CLI[@]}" train --data "${LOW[$m]}" ${MODE[$a]} "${RULE_C[@]}" "${BLOCKS[@]}" --epochs "$epochs" "${BUDGET[@]}" \
    --describe "DeepCAD-RT $m frames 0-1199 (validation 1210-1359), arm $a: brick 40" \
    --out "$OUT/$name.bin" --journey "$OUT/journey.json" > "$OUT/$name.train.log"
  tail -2 "$OUT/$name.train.log"
}
for m in "${MOVIES[@]}"; do for a in "${ARMS[@]}"; do train "$m" "$a" 60 "${m}_$a"; done; done

# 2. Convergence: validation loss at a run's last epoch more than 1% below its loss ten epochs earlier = not converged
#    (epochs 50 and 60 for the arms; 170 and 180 for the post-hoc long run).
val_loss() { awk -v e="$2" '$1 == "📊" && $2 == "epoch" && $3 == e { for (i = 1; i <= NF; i++) if ($i == "val") print $(i + 1) }' "$1"; }
converged() {   # prints 1 (converged) or 0
  local last before after
  last="$(awk '$1 == "📊" && $2 == "epoch" { e = $3 } END { print e }' "$1")"
  before="$(val_loss "$1" $((last - 10)))" after="$(val_loss "$1" "$last")"
  awk -v a="$before" -v b="$after" 'BEGIN { print (b < 0.99 * a) ? 0 : 1 }'
}
# The post-hoc diagnostic (pre-registered): if A3 isn't converged on a movie, one A3 run at 180 x 50 steps, otherwise
# identical, reported beside the arms and never read by the adoption rule.
for m in "${MOVIES[@]}"; do
  if [[ "$(converged "$OUT/${m}_A3.train.log")" == 0 ]]; then train "$m" A3 180 "${m}_A3_long"; fi
done

# 3. Scores: every arm (and the noisy input) on the test block; on the neutrophils also the late block.
score() {   # score <movie> <name> <block> <out>
  [[ -f "$4" ]] || "${CLI[@]}" seq-eval --stack "${LOW[$1]}" --reference "${HIGH[$1]}" --frames "$3" --weights "$OUT/$2.bin" \
    --per-frame "${4%.txt}.tsv" > "$4"
}
runs() { local m="$1"; for a in "${ARMS[@]}"; do echo "${m}_$a"; done; [[ -f "$OUT/${m}_A3_long.bin" ]] && echo "${m}_A3_long"; true; }
for m in "${MOVIES[@]}"; do
  for name in $(runs "$m"); do
    score "$m" "$name" "$TEST" "$EVAL/eval_${name}_test.txt"
    [[ "$m" == neutrophils ]] && score "$m" "$name" "$LATE" "$EVAL/eval_${name}_late.txt"
  done
done

# 4. The summary: one line per movie, block and run (unrounded to 4 decimals), and the noisy input's.
#   run <movie> <block> <arm> <PSNR> <SSIM> <dynamic> <steady> <slope> <ratio-err> <found> <real> <converged> <train-min>
{
  for m in "${MOVIES[@]}"; do
    for block in test late; do
      [[ "$block" == late && "$m" != neutrophils ]] && continue
      first=1
      for name in $(runs "$m"); do
        f="$EVAL/eval_${name}_$block.txt"
        if (( first )); then awk -v m="$m" -v b="$block" '$1 == "noisy" { print "noisy", m, b, $2, $3, $4, $5, $6, $7, $8, $9 }' "$f"; first=0; fi
        arm="${name#"${m}_"}"
        minutes="$(sed -n 's/.*epochs in \([0-9]*\):\([0-9]*\).*/\1 \2/p' "$OUT/$name.train.log" | awk '{ printf "%.1f", $1 + $2 / 60 }')"
        awk -v m="$m" -v b="$block" -v a="$arm" -v c="$(converged "$OUT/$name.train.log")" -v t="$minutes" \
          '$1 == "method" { print "run", m, b, a, $2, $3, $4, $5, $6, $7, $8, $9, c, t }' "$f"
      done
    done
  done
} > "$EVAL/summary.txt"

# 5. The pre-registered reading.
awk -f tools/experiments/timelapse_reading.awk "$EVAL/summary.txt" > "$EVAL/reading.txt"
cat "$EVAL/summary.txt" "$EVAL/reading.txt"
