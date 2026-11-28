#!/usr/bin/env bash
# Brick 38, the self-supervised scoreboard: DORA's default Noise2Void (routed since Q20) against CAREamics 0.3.5
# Noise2Void and N2V2 on every benchmark DORA can score here, pre-registered in PATHWAY.md (commit f91afa8) before any
# of its code existed. A measurement, not an adoption pass: its reading only orders the next bricks. Results in
# docs/RESULTS.md.
#
#   tools/fetch_fmd.sh && tools/fetch_ext.sh && tools/fetch_ext.sh mdc25-tribolium biosr-ccps
#   python3 -m venv VENV && VENV/bin/pip install -r tools/experiments/requirements-careamics.txt
#   CAREAMICS_PYTHON=VENV/bin/python tools/experiments/scoreboard.sh
#
# Every method gets the same budget: 30 x 50 steps, batch 16, 64 px patches, seed 2026, CPU. CAREamics Noise2Void on
# MreB, nucleoid and CARE 2D is Q18's run (runs/q18, same script and protocol), not re-run. Finished steps are skipped.
# Writes runs/scoreboard/summary.txt and the reading (runs/scoreboard/reading.txt, tools/experiments/scoreboard_reading.awk).
#
#   tools/experiments/scoreboard.sh --rescore DIR
# scores every finished run again into DIR (DIR/summary.txt, DIR/reading.txt) and trains nothing: gate 38 compares that
# summary with the one recorded in docs/RESULTS.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/scoreboard Q18=runs/q18 EVAL=runs/scoreboard RESCORE=0
if [[ "${1:-}" == --rescore ]]; then EVAL="${2:?--rescore needs a folder}" RESCORE=1; fi
mkdir -p "$OUT" "$EVAL"
dotnet build tools/DoraBio.Cli -c Release -v q >/dev/null
CLI=(dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll)
EXT=(deepbacs-mreb deepbacs-nucleoid zerocost-care2d mdc25-tribolium-2d biosr-ccps-2d)
Q18_SETS=" deepbacs-mreb deepbacs-nucleoid zerocost-care2d "
BUDGET=(--epochs 30 --steps 50 --batch 16 --patch 64 --seed 2026 --mode n2v --journey "$OUT/journey.json")

missing() { echo "--rescore: $1 is missing; run tools/experiments/scoreboard.sh first" >&2; exit 1; }

# 1. The derived 2D sets and FMD in the external sets' layout (for CAREamics).
(( RESCORE )) || for d in mdc25-tribolium-2d biosr-ccps-2d; do
  [[ -f "data/ext/$d/.complete" ]] || "${CLI[@]}" stack-export --dataset "$d"
done
(( RESCORE )) || [[ -f data/ext/fmd-raw/.complete ]] || "${CLI[@]}" fmd-export --root data/fmd --out data/ext/fmd-raw

# 2. DORA's default: `train --mode n2v` with nothing else, so the noise check routes (Q20).
train_dora() {   # train_dora <set> <data args...>
  local d="$1"; shift
  [[ -f "$OUT/dora_$d.bin" ]] && return 0
  (( RESCORE )) && missing "$OUT/dora_$d.bin"
  echo "=== DORA default Noise2Void on $d"
  "${CLI[@]}" train "$@" "${BUDGET[@]}" --out "$OUT/dora_$d.bin" > "$OUT/dora_$d.train.log"
  tail -1 "$OUT/dora_$d.train.log"
}
train_dora fmd --data data/fmd --fmd-frames 4 --fmd-levels raw --describe "FMD raw training frames, 4 per FOV: brick 38 scoreboard"
for d in "${EXT[@]}"; do
  train_dora "$d" --ext "$d" --data data/ext --describe "$d training split, noisy images only: brick 38 scoreboard"
done
dora_args() {   # dora_args <set>: --weights, plus --weights-correlated when the set trained two networks
  local args=(--weights "$OUT/dora_$1.bin")
  [[ -f "$OUT/dora_$1.correlated.bin" ]] && args+=(--weights-correlated "$OUT/dora_$1.correlated.bin")
  echo "${args[@]}"
}

# 3. CAREamics Noise2Void (Q18's runs where they exist) and N2V2.
careamics() {   # careamics <variant n2v|n2v2> <set>
  local v="$1" d="$2" flag=()
  [[ "$v" == n2v2 ]] && flag=(--n2v2)
  [[ -f "$OUT/careamics_$v/$d/timing.json" ]] && return 0
  (( RESCORE )) && missing "$OUT/careamics_$v/$d"
  local PY="${CAREAMICS_PYTHON:?set CAREAMICS_PYTHON to a python with tools/experiments/requirements-careamics.txt installed}"
  echo "=== CAREamics $v on $d"
  "$PY" tools/experiments/ext_careamics.py --algorithm n2v "${flag[@]}" --dataset "$d" --root data/ext --out "$OUT" > "$OUT/careamics_${v}_$d.log" 2>&1
  tail -1 "$OUT/careamics_${v}_$d.log"
}
n2v_dir() { [[ "$Q18_SETS" == *" $1 "* ]] && echo "$Q18/careamics_n2v/$1" || echo "$OUT/careamics_n2v/$1"; }
for d in fmd-raw "${EXT[@]}"; do
  if [[ "$Q18_SETS" == *" $d "* ]]; then [[ -f "$Q18/careamics_n2v/$d/timing.json" ]] || missing "$Q18/careamics_n2v/$d (tools/experiments/q18_ext.sh)"
  else careamics n2v "$d"; fi
  careamics n2v2 "$d"
done

# 4. Scores: FMD with its own protocol, the others with CARE's; MreB's fidelity.
LABEL_N2V="CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU"
LABEL_N2V2="CAREamics 0.3.5 N2V2, default 2D config, 30 x 50 steps x batch 16, CPU"
run() { local f="$1"; shift; [[ -f "$f" ]] || "${CLI[@]}" "$@" > "$f"; }
run "$EVAL/eval_dora_fmd.txt" fmd-eval $(dora_args fmd) --root data/fmd --levels raw --precise
run "$EVAL/eval_n2v_fmd.txt" fmd-eval --predictions "$OUT/careamics_n2v/fmd-raw" --label "$LABEL_N2V" --root data/fmd --precise
run "$EVAL/eval_n2v2_fmd.txt" fmd-eval --predictions "$OUT/careamics_n2v2/fmd-raw" --label "$LABEL_N2V2" --root data/fmd --precise
for d in "${EXT[@]}"; do
  run "$EVAL/eval_dora_$d.txt" ext-eval --dataset "$d" $(dora_args "$d") --precise
  run "$EVAL/eval_n2v_$d.txt" ext-eval --dataset "$d" --predictions "$(n2v_dir "$d")" --label "$LABEL_N2V" --precise
  run "$EVAL/eval_n2v2_$d.txt" ext-eval --dataset "$d" --predictions "$OUT/careamics_n2v2/$d" --label "$LABEL_N2V2" --precise
done
run "$EVAL/fid_dora_deepbacs-mreb.txt" fidelity --dataset deepbacs-mreb $(dora_args deepbacs-mreb)
run "$EVAL/fid_n2v_deepbacs-mreb.txt" fidelity --dataset deepbacs-mreb --predictions "$(n2v_dir deepbacs-mreb)" --label "$LABEL_N2V"
run "$EVAL/fid_n2v2_deepbacs-mreb.txt" fidelity --dataset deepbacs-mreb --predictions "$OUT/careamics_n2v2/deepbacs-mreb" --label "$LABEL_N2V2"

# 5. The summary: one line per set (unrounded PSNR, then SSIM), BioSR+ per signal level, fidelity, training time.
seconds() { awk -F: '{ print $1 * 60 + $2 }' <<< "$1"; }
{
  # set <name> <noisy> <bilateral> <DORA> <N2V> <N2V2> PSNR, then the same five SSIMs
  f() { awk '$1 == "all," { for (i = 1; i <= NF; i++) if ($i == "noisy" || $i == "bilateral" || $i == "DORA" || $i == "method") v[$i] = $(i + 1) }
             END { printf "%s %s %s", v["noisy"], v["bilateral"], (v["DORA"] != "" ? v["DORA"] : v["method"]) }' "$1"; }
  fs() { awk '$1 == "all" { print $5, $8, $11 }' "$1"; }
  read -r n b dora <<< "$(f "$EVAL/eval_dora_fmd.txt")"; read -r _ _ n2v <<< "$(f "$EVAL/eval_n2v_fmd.txt")"; read -r _ _ n2v2 <<< "$(f "$EVAL/eval_n2v2_fmd.txt")"
  read -r sn sb sd <<< "$(fs "$EVAL/eval_dora_fmd.txt")"; read -r _ _ s1 <<< "$(fs "$EVAL/eval_n2v_fmd.txt")"; read -r _ _ s2 <<< "$(fs "$EVAL/eval_n2v2_fmd.txt")"
  echo "set fmd $n $b $dora $n2v $n2v2 $sn $sb $sd $s1 $s2"
  for m in Confocal TwoPhoton WideField; do
    echo "fmd-modality $m $(for e in dora n2v n2v2; do awk -v m="$m" '$1 == m { print $9 }' "$EVAL/eval_${e}_fmd.txt"; done | paste -sd' ')"
  done
  for d in "${EXT[@]}"; do
    p() { awk '$1 == "mean," { print $4, $7, $10 }' "$1"; }
    s() { awk '$1 == "mean" { print $4, $7, $10 }' "$1"; }
    read -r n b dora <<< "$(p "$EVAL/eval_dora_$d.txt")"; read -r _ _ n2v <<< "$(p "$EVAL/eval_n2v_$d.txt")"; read -r _ _ n2v2 <<< "$(p "$EVAL/eval_n2v2_$d.txt")"
    read -r sn sb sd <<< "$(s "$EVAL/eval_dora_$d.txt")"; read -r _ _ s1 <<< "$(s "$EVAL/eval_n2v_$d.txt")"; read -r _ _ s2 <<< "$(s "$EVAL/eval_n2v2_$d.txt")"
    echo "set $d $n $b $dora $n2v $n2v2 $sn $sb $sd $s1 $s2"
  done
  # BioSR+ per signal level (mean of the 2-decimal per-image PSNRs; reported, not part of the reading)
  for lvl in 1 2 3 4 5 6 7 8; do
    printf "biosr-level %d" "$lvl"
    for e in dora n2v n2v2; do
      awk -v l="__L0$lvl" -v e="$e" 'index($1, l) && $1 != "mean" { n++; a += $2; b += $5; m += $8 }
        END { if (e == "dora") printf " %.2f %.2f", a / n, b / n; printf " %.2f", m / n }' "$EVAL/eval_${e}_biosr-ccps-2d.txt"
    done
    echo
  done
  for e in dora n2v n2v2; do echo "mreb-fidelity $e $(awk '$1 == "method" { print $2, $3, $4, $5 }' "$EVAL/fid_${e}_deepbacs-mreb.txt")"; done
  for d in fmd "${EXT[@]}"; do
    c="$d"; [[ "$d" == fmd ]] && c=fmd-raw
    t1="$(sed -n 's/.*routed training done in \([0-9:]*\).*/\1/p' "$OUT/dora_$d.train.log")"
    n2v_t="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_seconds"])' "$(n2v_dir "$c")/timing.json")"
    n2v2_t="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_seconds"])' "$OUT/careamics_n2v2/$c/timing.json")"
    echo "train-seconds $d $(seconds "$t1") $n2v_t $n2v2_t"
  done
} > "$EVAL/summary.txt"

# 6. The pre-registered reading.
awk -f tools/experiments/scoreboard_reading.awk "$EVAL/summary.txt" > "$EVAL/reading.txt"
cat "$EVAL/summary.txt" "$EVAL/reading.txt"
