#!/usr/bin/env bash
# Brick-by-brick build for DORA v2.0-bio.
#
# Each brick sits on the ones below it. A brick is "laid" only when its own gate AND every gate
# beneath it pass. Run the gates in order; the run stops at the first red one.
#
#   ./bricks.sh             all gates, in order
#   ./bricks.sh 4           gates 0..4 (everything up to and including brick 4)
#   ./bricks.sh --only 5    just gate 5 (use while iterating on that brick; finish with a full run)
#   ./bricks.sh --list      show the bricks and what each gate checks
#   ./bricks.sh --self-test plant a known bug per gate in a scratch copy (data excluded); each must turn its gate red
#
# Gate 8 needs the Godot 4.5 .NET binary: GODOT=/path/to/Godot_v4.5.x-stable_mono_linux.x86_64 ./bricks.sh
# Without it, gate 8 is reported as SKIPPED. That's not a pass, and the summary says so.
# REQUIRE_GODOT=1 turns a missing Godot into a FAIL (CI sets it, so CI can never "pass" by skipping).
# Gates 9-10 need python3 with PyYAML (to check the CI workflow).
# Gates 11-13 check the real FMD dataset (tools/fetch_fmd.sh). Without the data their data-dependent part
# reports SKIPPED; the parts that need no data (checksum pins, loader unit tests, recorded results) still run.
# Any gate may report SKIPPED (exit code 77, Automake's convention; never used by the tools for errors), but only
# for a check it can't run here. That is never a pass. Every other non-zero exit is a FAIL.
set -uo pipefail
cd "$(dirname "$0")"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1

NAMES=(scaffold nn data io inference training cli godot-build godot-smoke ci-core ci-godot fmd-fetch fmd-loader fmd-eval)
CHECKS=(
  "solution builds (warnings are errors, missing XML docs included); every test class belongs to exactly one brick"
  "ConvLayer/DenoisingNet/Adam/Simd: finite-difference gradient checks, optimizer behavior"
  "GrayImage, Normalization, Dihedral, BlindSpot, PatchSampler, Synthetic: data invariants"
  "TIFF/PNG decoders vs Pillow/ImageMagick fixtures, TIFF round-trip, weights file + CRC"
  "Metrics (PSNR/SSIM/noise σ), exact tiled inference, bilateral baseline"
  "Trainer, validation, progressive targets, journey log, online adapter, dataset loader"
  "dora-bio CLI end to end: synth → train → info → denoise → eval, plus the bundled model and sample"
  "Godot C# assembly compiles against Godot.NET.Sdk 4.5.1; every member in godot/scripts/*.cs is documented"
  "headless Godot: Main.tscn loads, sample → denoise → adapt, main thread never blocked"
  "CI workflow runs the full gate set on every dora-bio change; --self-test proves gates 0-7 can go red"
  "CI Godot job pins the SDK's Godot version, verifies its SHA-512, and makes gate 8 mandatory"
  "FMD fetch script pins every archive (Drive id, size, SHA-256); downloaded data matches the pins"
  "FMD loader unit tests (real layouts, test-FOV exclusion); with data: counts + no train/test leakage"
  "docs/RESULTS.md exists; with data: the recorded synthetic-model evaluation reproduces exactly"
)
LAST=$((${#NAMES[@]} - 1))
LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dora-bricks.XXXXXX")"
trap 'rm -rf "$LOG_DIR"' EXIT

list() {
  for i in $(seq 0 $LAST); do printf "  %d  %-12s %s\n" "$i" "${NAMES[$i]}" "${CHECKS[$i]}"; done
}

# Runs one brick's xunit tests; fails if any test fails OR if the filter matched nothing.
test_gate() {
  local brick="$1" out
  out="$(dotnet test DoraBio.sln -c Release --filter "Brick=$brick" 2>&1)"; local code=$?
  echo "$out"
  local passed
  passed="$(echo "$out" | sed -n 's/.*Passed: *\([0-9][0-9]*\).*/\1/p' | tail -1)"
  [[ $code -eq 0 && -n "$passed" && "$passed" -gt 0 ]] || { echo ">> gate needs at least one passing test and no failures"; return 1; }
  echo ">> $passed tests passed"
}

gate_0() { dotnet build DoraBio.sln -c Release && test_gate 0-scaffold; }
gate_1() { test_gate 1-nn; }
gate_2() { test_gate 2-data; }
gate_3() { test_gate 3-io; }
gate_4() { test_gate 4-inference; }
gate_5() { test_gate 5-training; }

gate_6() (   # subshell body: `set -e` aborts only this gate, never the runner
  set -e
  dotnet build tools/DoraBio.Cli -c Release
  cli="dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll" t="$LOG_DIR/cli"
  $cli synth --out "$t/data" --count 4 --size 64 --seed 5
  $cli train --data "$t/data" --out "$t/w.bin" --mode n2v --epochs 2 --steps 3 --batch 4 --patch 32 \
             --features 8 --journey "$t/journey.json" --describe "bricks.sh gate 6 smoke data"
  $cli info --weights "$t/w.bin" | grep -q "bricks.sh gate 6 smoke data"
  $cli denoise --weights "$t/w.bin" --in "$t/data/noisy/synthetic_000.tif" --out "$t/out.tif" --baseline "$t/base.tif"
  $cli eval --weights "$t/w.bin" --data "$t/data"
  test -s "$t/out.tif" && test -s "$t/base.tif"
  grep -q '"Status": "completed"' "$t/journey.json"
  # The committed artifacts must still load and run.
  $cli info --weights godot/models/dora_bio_weights.bin | grep -q "SYNTHETIC"
  $cli denoise --weights godot/models/dora_bio_weights.bin --in godot/samples/synthetic_cells.tif --out "$t/sample.tif"
)

gate_7() {
  dotnet build godot/DoraBio.Godot.csproj || return 1
  # Our C# in godot/scripts must be fully documented. Godot's source generators emit undocumented public
  # members (EmitSignal…), so the compiler can't enforce this; filter to our files instead.
  local missing
  missing="$(dotnet build godot/DoraBio.Godot.csproj -p:GenerateDocumentationFile=true --no-incremental 2>&1 \
    | grep -E "warning CS(1591|157[0-4])" | grep "/godot/scripts/" | sort -u)"
  if [[ -n "$missing" ]]; then echo "$missing"; echo ">> undocumented members in godot/scripts"; return 1; fi
}

gate_8() {
  local godot="${GODOT:-$(command -v godot || true)}"
  if [[ -z "$godot" || ! -x "$godot" ]]; then
    echo "Godot 4.5 .NET binary not found; set GODOT=/path/to/Godot_v4.5.x-stable_mono_..."
    [[ "${REQUIRE_GODOT:-0}" == "1" ]] && { echo ">> REQUIRE_GODOT=1: a missing Godot is a failure"; return 1; }
    return 77
  fi
  gate_7 >/dev/null || return 1
  (cd godot && timeout 300 "$godot" --headless --path . --import >/dev/null 2>&1)
  local out
  out="$(cd godot && timeout 300 "$godot" --headless --path . res://tests/smoke_test.tscn 2>&1)"; local code=$?
  echo "$out"
  [[ $code -eq 0 ]] && echo "$out" | grep -q "SMOKE TEST PASSED"
}

WORKFLOW=../.github/workflows/dora-bio.yml

gate_9() {
  python3 tools/ci/check_workflow.py "$WORKFLOW" core || return 1
  self_test
}

gate_10() { python3 tools/ci/check_workflow.py "$WORKFLOW" godot; }

FMD=data/fmd
CLI_DLL=tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll

gate_11() {   # fmd-fetch
  bash -n tools/fetch_fmd.sh || return 1
  local entries pinned
  entries="$(grep -cE '^  "[A-Za-z_]+\|' tools/fetch_fmd.sh)"
  pinned="$(grep -cE '^  "[A-Za-z_]+\|[A-Za-z0-9_-]+\|[0-9]+\|[0-9a-f]{64}"$' tools/fetch_fmd.sh)"
  [[ "$entries" -eq 13 && "$pinned" -eq 13 ]] || { echo ">> $pinned of $entries archives fully pinned (need 13 with id, size and sha256)"; return 1; }
  [[ -d "$FMD" ]] || { echo "pins OK; data check skipped (no FMD data: run tools/fetch_fmd.sh)"; return 77; }
  local missing=0 name sha want got count
  while IFS='|' read -r name _ _ sha; do
    sha="${sha%\"}"
    if [[ ! -f "$FMD/$name/.complete" ]]; then missing=$((missing + 1)); continue; fi
    got="$(cat "$FMD/$name/.complete")"
    [[ "$got" == "$sha" ]] || { echo ">> $name was extracted from an archive with sha256 $got, pinned $sha"; return 1; }
    want=$([[ "$name" == test_mix ]] && echo 288 || echo 5020)
    count="$(find "$FMD/$name" -type f -name '*.png' | wc -l)"
    [[ "$count" -eq "$want" ]] || { echo ">> $name has $count images, expected $want"; return 1; }
  done < <(grep -E '^  "[A-Za-z_]+\|' tools/fetch_fmd.sh | sed 's/^  "//')
  [[ $missing -eq 0 ]] || { echo "present archives verified; $missing not downloaded (run tools/fetch_fmd.sh)"; return 77; }
  echo ">> all 13 archives verified against their pins"
}

gate_12() {   # fmd-loader
  test_gate 12-fmd || return 1
  [[ -f "$FMD/test_mix/.complete" ]] || { echo "unit tests passed; real-data check skipped (no FMD data)"; return 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  dotnet "$CLI_DLL" fmd-check --root "$FMD"
}

gate_13() {   # fmd-eval
  [[ -f docs/RESULTS.md ]] || { echo ">> docs/RESULTS.md is missing"; return 1; }
  [[ -f "$FMD/test_mix/.complete" ]] || { echo "results recorded; reproduction skipped (no FMD data)"; return 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  # The block between the markers in RESULTS.md is the verbatim output of this command; it must reproduce exactly.
  local recorded fresh
  recorded="$(sed -n '/<!-- fmd-eval:synthetic -->/,/<!-- \/fmd-eval -->/p' docs/RESULTS.md | sed '1,2d;$d' | sed '$d')"
  fresh="$(dotnet "$CLI_DLL" fmd-eval --weights godot/models/dora_bio_weights.bin --root "$FMD" --levels raw)"
  [[ -n "$recorded" ]] || { echo ">> no <!-- fmd-eval:synthetic --> block in docs/RESULTS.md"; return 1; }
  if [[ "$recorded" != "$fresh" ]]; then
    echo ">> recorded synthetic-model results differ from a fresh run:"; diff <(echo "$recorded") <(echo "$fresh") | head -20; return 1
  fi
  echo ">> recorded results reproduce exactly"
}

# ---- gate self-test ---------------------------------------------------------
# A gate that can't fail proves nothing. For each of gates 0-7, plant one known bug in a fresh scratch copy of
# this folder and check that exactly that gate turns red. Mutations are exact string replacements that must
# match once. If the code moves on and a mutation no longer applies, the self-test fails and says so.
MUTATIONS=(
  # gate | file | text to find | replacement ("<DELETE-FILE>" deletes the file)
  $'0|tests/DoraBio.Core.Tests/DataTests.cs|[Trait(Bricks.Trait, Bricks.Data)]\n|'
  $'0|src/DoraBio.Core/Data/GrayImage.cs|    /// <summary>Width in pixels (number of columns).</summary>\n|'
  '1|src/DoraBio.Core/Nn/ConvLayer.cs|gradients[BiasOffset + oc] += Simd.Sum(gOutPlane);|gradients[BiasOffset + oc] += 1.1f * Simd.Sum(gOutPlane);'
  '2|src/DoraBio.Core/Data/Dihedral.cs|(a, b) = (b, last - a);|(a, b) = (b, a);'
  '3|src/DoraBio.Core/IO/PngDecoder.cs|3 => (a + b) >> 1,|3 => (a + b + 1) >> 1,'
  '4|src/DoraBio.Core/TiledInference.cs|margin = net.Config.ReceptiveRadius;|margin = net.Config.ReceptiveRadius - 1;'
  '5|src/DoraBio.Core/Training/Trainer.cs|old * cfg.LearningRateDecay|old / cfg.LearningRateDecay'
  '6|godot/samples/synthetic_cells.tif||<DELETE-FILE>'
  $'7|godot/scripts/DoraBioBridge.cs|    /// <summary>True while a denoise, adapt or training job runs. Only one job runs at a time.</summary>\n|'
  '11|tools/fetch_fmd.sh|d7a14d3be17368dbfe24a5cfe8ed0d40d403d7902d70fe914d04303e4aa8f2f6"|"'
)

self_test() {
  local root failures=0 n=0
  root="$(mktemp -d "${TMPDIR:-/tmp}/dora-selftest.XXXXXX")"

  # Control: in an unmutated scratch copy, every gate that has a planted bug must be green, or SKIPPED for want
  # of data. Otherwise a "caught" result could come from a broken copy rather than from the planted bug, and the
  # self-test would prove nothing. (Gate 9 itself is never run in a copy, which would recurse.)
  mkdir -p "$root/control"
  tar --exclude=bin --exclude=obj --exclude=.godot --exclude=./data --exclude=./runs -cf - . | tar -xf - -C "$root/control"
  local -A seen=()
  local gates=() g
  for m in "${MUTATIONS[@]}"; do g="${m%%|*}"; [[ -n "${seen[$g]:-}" ]] || { seen[$g]=1; gates+=("$g"); }; done
  for g in "${gates[@]}"; do
    if (cd "$root/control" && ./bricks.sh --only "$g" 2>&1 | grep -q "▶ brick $g .*FAIL"); then
      echo "control (no mutation): gate $g is red in a scratch copy; mutation results would be meaningless"
      rm -rf "$root"
      return 1
    fi
  done
  echo "control (no mutation): gates ${gates[*]} not red"

  for m in "${MUTATIONS[@]}"; do
    IFS='|' read -r -d '' gate file old new <<<"$m" || true
    new="${new%$'\n'}"   # here-strings append a newline
    n=$((n + 1))
    local copy="$root/m$n"
    mkdir -p "$copy"
    tar --exclude=bin --exclude=obj --exclude=.godot --exclude=./data --exclude=./runs -cf - . | tar -xf - -C "$copy"

    if [[ "$new" == "<DELETE-FILE>" ]]; then
      rm "$copy/$file" || { echo "mutation $n: $file not found"; failures=$((failures + 1)); continue; }
    elif ! OLD="$old" NEW="$new" python3 - "$copy/$file" <<'PY'
import os, sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
old, new = os.environ["OLD"], os.environ["NEW"]
if text.count(old) != 1:
    sys.exit(f"found {text.count(old)} times (need exactly 1)")
open(path, "w", encoding="utf-8").write(text.replace(old, new))
PY
    then
      echo "mutation $n (gate $gate, $file): DID NOT APPLY; update MUTATIONS in bricks.sh"
      failures=$((failures + 1)); continue
    fi

    local out
    out="$(cd "$copy" && ./bricks.sh --only "$gate" 2>&1)"
    if echo "$out" | grep -q "▶ brick $gate .*FAIL"; then
      echo "mutation $n (gate $gate, $file): caught"
    else
      echo "mutation $n (gate $gate, $file): NOT CAUGHT, gate $gate stayed green"
      failures=$((failures + 1))
    fi
  done
  rm -rf "$root"
  [[ $failures -eq 0 ]] && echo ">> all $n planted bugs caught"
  return $((failures > 0))
}

# ---- argument handling ------------------------------------------------------
FIRST=0; UPTO=$LAST
case "${1:-}" in
  --list) list; exit 0 ;;
  --self-test) self_test; exit $? ;;
  --only) [[ "${2:-}" =~ ^[0-9]+$ && $2 -le $LAST ]] || { echo "usage: $0 --only N (0..$LAST)"; exit 64; }; FIRST=$2; UPTO=$2 ;;
  "") ;;
  *) [[ "$1" =~ ^[0-9]+$ && $1 -le $LAST ]] || { echo "usage: $0 [N | --only N | --list]"; exit 64; }; UPTO=$1 ;;
esac

declare -a SUMMARY=()
STATUS=0
for i in $(seq "$FIRST" "$UPTO"); do
  log="$LOG_DIR/brick$i.log"
  printf "▶ brick %d %-12s " "$i" "${NAMES[$i]}"
  start=$SECONDS
  "gate_$i" >"$log" 2>&1; code=$?
  took=$((SECONDS - start))
  if [[ $code -eq 0 ]]; then
    printf "PASS  (%ss)  %s\n" "$took" "$(grep -oE '>> ([0-9]+ tests passed|all [0-9]+ planted bugs caught)' "$log" | sed 's/>> //' || true)"
    SUMMARY+=("$i ${NAMES[$i]}: PASS")
  elif [[ $code -eq 77 ]]; then
    printf "SKIPPED  %s\n" "$(tail -1 "$log")"
    SUMMARY+=("$i ${NAMES[$i]}: SKIPPED (not verified)")
  else
    printf "FAIL  (%ss)\n" "$took"
    echo "──── last 40 lines of brick $i log ────"
    tail -40 "$log"
    echo "───────────────────────────────────────"
    echo "Stopped at brick $i. Fix it before laying anything on top."
    STATUS=1
    break
  fi
done

echo
printf '%s\n' "${SUMMARY[@]}"
if [[ $STATUS -ne 0 ]]; then echo "RESULT: RED"; elif printf '%s\n' "${SUMMARY[@]}" | grep -q SKIPPED; then echo "RESULT: GREEN with skipped gates"; else echo "RESULT: GREEN"; fi
exit $STATUS
