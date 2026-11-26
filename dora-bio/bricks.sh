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

NAMES=(scaffold nn data io inference training cli godot-build godot-smoke ci-core ci-godot fmd-fetch fmd-loader fmd-eval
       n2v-diagnostic structured-mask vst calibration pool-upsample unet unet-tiling arch-eval
       bench blocked-conv perf-budget q1-recipe tiff-stack-write q10-routing cli-batch q13-conditioning ext-data q18-ext-bench quant-fidelity)
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
  "benchmark unit tests; docs/RESULTS.md exists; with data: the recorded synthetic-model evaluation and noise measurement reproduce exactly"
  "correlated-noise check from the noisy image alone; with data: agrees with the ground-truth measurement on every raw test image"
  "structured blind spots; recorded FMD results meet the bar set in advance (widefield +1 dB, others -0.2 dB at worst); with data + weights: they reproduce"
  "generalized Anscombe transform: unit variance across intensities on synthetic data, exact unbiased inverse"
  "noise model (variance = gain · intensity + intercept) from one image: synthetic gain within 10%, variance curve within 10%"
  "2×2 average pooling and nearest upsampling: finite-difference gradients, adjoint identities"
  "U-Net: whole-network finite-difference gradients; weights format 2 round-trips both architectures; format-1 files still load"
  "U-Net tiled inference equals whole-image inference for ragged sizes (tiles on the 2^depth grid)"
  "recorded FMD results for both architectures reproduce; the default follows the rule set before measuring"
  "dora-bio bench: denoise and train-step timings for both architectures, as JSON that passes the schema check"
  "register-blocked convolution: bit-identical to the reference kernels (forward, weight and input gradients, whole nets)"
  "dora-bio bench stays within the time budgets in tools/ci/perf_budget.json"
  "quality pass Q1: the recorded verdict follows the pre-registered rule; with data + weights the blocks reproduce"
  "multi-page TIFF stacks round-trip in every sample type; Pillow (an independent reader) reads a written stack exactly"
  "quality pass Q10: router unit tests; the recorded verdict follows the pre-registered rule; with data + weights the blocks reproduce"
  "stacks and folders stream one frame at a time: pages decode exactly, no frame outlives the next; CLI denoises a stack and a folder; peak memory flat in stack length"
  "quality pass Q13: FiLM gradients, untrained = plain bit for bit, exact tiling; the recorded verdict and the default follow the pre-registered rule; with data + weights the blocks reproduce"
  "external datasets: fetch_ext.sh pins every archive (record, size, sha256); pairing by name; with data: ext-check consistent"
  "quality pass Q18: CARE's scoring protocol; the recorded verdict follows the pre-registered rule; with data + runs/q18 the blocks reproduce"
  "fidelity (proportionality, ratio error, spots found/real) exact on synthetic ground truth; with data + weights the recorded blocks reproduce"
)
LAST=$((${#NAMES[@]} - 1))
LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dora-bricks.XXXXXX")"
trap 'rm -rf "$LOG_DIR"' EXIT

list() {
  for i in $(seq 0 $LAST); do printf "  %d  %-15s %s\n" "$i" "${NAMES[$i]}" "${CHECKS[$i]}"; done
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
EXT=data/ext
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
  test_gate 13-fmd-eval || return 1
  [[ -f docs/RESULTS.md ]] || { echo ">> docs/RESULTS.md is missing"; return 1; }
  [[ -f "$FMD/test_mix/.complete" ]] || { echo "results recorded; reproduction skipped (no FMD data)"; return 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  reproduces fmd-eval-synthetic fmd-eval --weights godot/models/dora_bio_weights.bin --root "$FMD" --levels raw || return 1
  reproduces fmd-noise fmd-noise --root "$FMD" --levels raw || return 1
}

gate_14() {   # n2v-diagnostic
  test_gate 14-n2v-diagnostic || return 1
  [[ -f "$FMD/test_mix/.complete" ]] || { echo "unit tests passed; real-data check skipped (no FMD data)"; return 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  # The blind check (noisy image only) must give the same verdict as the ground-truth measurement on every image.
  dotnet "$CLI_DLL" fmd-diagnose --root "$FMD" --levels raw
}

gate_15() {   # structured-mask
  test_gate 15-structured-mask || return 1
  # The bar was set before building (PATHWAY.md, P3) and never changed: on FMD test_mix (raw), vs one plain Noise2Void
  # network, widefield must gain at least 1 dB and confocal and two-photon lose at most 0.2 dB. The first design (one
  # network, structured blind spots on the flagged images) missed it and stays recorded as p3-n2v-shared; the bar is
  # checked on the second design, two networks routed by the noise check (p3-n2v-routed).
  local m plain routed bar
  for m in WideField Confocal TwoPhoton; do
    plain="$(recorded_block p3-n2v-plain | awk -v m="$m" '$1 == m { print $9 }')"
    routed="$(recorded_block p3-n2v-routed | awk -v m="$m" '$1 == m { print $9 }')"
    [[ -n "$plain" && -n "$routed" ]] || { echo ">> docs/RESULTS.md needs p3-n2v-plain and p3-n2v-routed fmd-eval blocks"; return 1; }
    bar=$([[ "$m" == WideField ]] && echo 1.0 || echo -0.2)
    awk -v a="$routed" -v b="$plain" -v n="$bar" 'BEGIN { exit !(a - b >= n - 1e-9) }' \
      || { echo ">> $m: routed $routed dB vs plain $plain dB misses the bar ($bar dB)"; return 1; }
    echo ">> $m: routed $routed dB vs plain $plain dB, bar $bar dB: met"
  done
  local w=runs/p3
  [[ -f "$FMD/test_mix/.complete" && -f $w/n2v_plain.bin && -f $w/n2v_struct.bin && -f $w/n2v_indep.bin && -f $w/n2v_corr.bin ]] \
    || { echo "bar met by the recorded results; reproduction skipped (needs FMD data and the weights from tools/experiments/p3_struct.sh)"; return 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  reproduces p3-n2v-plain fmd-eval --weights $w/n2v_plain.bin --root "$FMD" --levels raw || return 1
  reproduces p3-n2v-shared fmd-eval --weights $w/n2v_struct.bin --root "$FMD" --levels raw || return 1
  reproduces p3-n2v-routed fmd-eval --weights $w/n2v_indep.bin --weights-correlated $w/n2v_corr.bin --root "$FMD" --levels raw || return 1
}

gate_16() { test_gate 16-vst; }           # vst
gate_17() { test_gate 17-calibration; }   # calibration
gate_18() { test_gate 18-pool-upsample; } # pool-upsample
gate_19() { test_gate 19-unet; }          # unet
gate_20() { test_gate 20-unet-tiling; }   # unet-tiling

gate_22() (   # bench (subshell: set -e stops only this gate)
  set -e
  test_gate 22-bench
  dotnet build tools/DoraBio.Cli -c Release >/dev/null
  dotnet "$CLI_DLL" bench --quick --json "$LOG_DIR/bench.json"
  python3 tools/ci/check_bench.py "$LOG_DIR/bench.json"
)

gate_23() { test_gate 23-blocked-conv; }   # blocked-conv

gate_24() (   # perf-budget
  set -e
  test_gate 24-perf-budget
  dotnet build tools/DoraBio.Cli -c Release >/dev/null
  dotnet "$CLI_DLL" bench --json "$LOG_DIR/bench-full.json"
  python3 tools/ci/check_bench.py "$LOG_DIR/bench-full.json" --budget tools/ci/perf_budget.json
)

# levels_all: "level PSNR" pairs from the all-48 rows of a multi-level fmd-eval block on stdin.
levels_all() { awk '/^test_mix level/ { l = $3; sub(":", "", l) } $1 == "all" { print l, $9 }'; }

gate_25() {   # q1-recipe (quality pass Q1, pre-registered in PATHWAY.md before any run)
  # Rule: at raw, candidate >= baseline - 0.1 dB; mean over avg2..avg16 >= baseline mean + 0.5 dB; no lower level
  # below baseline - 0.1 dB. Adopted: supervised FMD training defaults to all noise levels. Not adopted: nothing ships.
  local base_raw expected recorded
  base_raw="$(recorded_block p4-unet-sup | awk '$1 == "all" { print $9 }')"
  [[ -n "$base_raw" && -n "$(recorded_block q1-baseline)" && -n "$(recorded_block q1-candidate)" ]] \
    || { echo ">> docs/RESULTS.md needs the p4-unet-sup, q1-baseline and q1-candidate blocks"; return 1; }
  expected="$( { echo "base raw $base_raw"; recorded_block q1-baseline | levels_all | sed 's/^/base /';
                 recorded_block q1-candidate | levels_all | sed 's/^/cand /'; } | awk '
    { v[$1 "," $2] = $3 }
    END {
      ok = v["cand,raw"] >= v["base,raw"] - 0.1 - 1e-9; split("avg2 avg4 avg8 avg16", L, " "); sum = 0
      for (i = 1; i <= 4; i++) { d = v["cand," L[i]] - v["base," L[i]]; sum += d; if (d < -0.1 - 1e-9) ok = 0 }
      if (sum / 4 < 0.5 - 1e-9) ok = 0
      printf "%s raw %+.2f, lower levels %+.2f on average\n", ok ? "adopted" : "not-adopted", v["cand,raw"] - v["base,raw"], sum / 4
    }')"
  recorded="$(sed -n 's/.*<!-- q1-verdict: \([a-z-]*\) -->.*/\1/p' docs/RESULTS.md)"
  [[ "${expected%% *}" == "$recorded" ]] || { echo ">> Q1's rule gives '${expected%% *}' ($expected), but RESULTS.md records '$recorded'"; return 1; }
  echo ">> Q1: $expected; recorded verdict agrees"
  local w=runs/q1
  [[ -f "$FMD/test_mix/.complete" && -f runs/p4/unet_sup.bin && -f $w/unet_levels60.bin && -f $w/unet_levels30.bin ]] \
    || { echo "verdict checked on the recorded results; reproduction skipped (needs FMD data and the Q1 weights)"; return 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  reproduces q1-baseline fmd-eval --weights runs/p4/unet_sup.bin --root "$FMD" --levels avg2,avg4,avg8,avg16 || return 1
  reproduces q1-candidate fmd-eval --weights $w/unet_levels60.bin --root "$FMD" --levels raw,avg2,avg4,avg8,avg16 || return 1
  reproduces q1-ablation fmd-eval --weights $w/unet_levels30.bin --root "$FMD" --levels raw,avg2,avg4,avg8,avg16 || return 1
}

gate_26() (   # tiff-stack-write (subshell: set -e stops only this gate)
  set -e
  test_gate 26-tiff-stack-write
  python3 -I -c "import PIL" 2>/dev/null || { echo "unit tests passed; independent-reader check skipped (Pillow not installed)"; exit 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null
  local t="$LOG_DIR/stack"
  dotnet "$CLI_DLL" synth --out "$t" --count 3 --width 37 --height 21 --seed 4 --stack "$t/stack.tif"
  python3 -I tools/ci/check_tiff_stack.py "$t/stack.tif" "$t/noisy"
)

gate_27() {   # q10-routing (quality pass Q10, pre-registered in PATHWAY.md before any of its code existed)
  test_gate 27-q10-routing || return 1
  # Rule: routed at raw >= A's raw (p4-unet-sup) - 0.1 dB, and routed mean over avg2..avg16 >= B's (q1-candidate) - 0.2 dB.
  local a_raw expected recorded
  a_raw="$(recorded_block p4-unet-sup | awk '$1 == "all" { print $9 }')"
  [[ -n "$a_raw" && -n "$(recorded_block q1-candidate)" && -n "$(recorded_block q10-routed)" ]] \
    || { echo ">> docs/RESULTS.md needs the p4-unet-sup, q1-candidate and q10-routed blocks"; return 1; }
  expected="$( { echo "a raw $a_raw"; recorded_block q1-candidate | levels_all | sed 's/^/b /';
                 recorded_block q10-routed | levels_all | sed 's/^/r /'; } | awk '
    { v[$1 "," $2] = $3 }
    END {
      split("avg2 avg4 avg8 avg16", L, " "); rb = 0; bb = 0
      for (i = 1; i <= 4; i++) { rb += v["r," L[i]] / 4; bb += v["b," L[i]] / 4 }
      ok = v["r,raw"] >= v["a,raw"] - 0.1 - 1e-9 && rb >= bb - 0.2 - 1e-9
      printf "%s raw %+.2f vs A, lower levels %+.2f vs B\n", ok ? "adopted" : "not-adopted", v["r,raw"] - v["a,raw"], rb - bb
    }')"
  recorded="$(sed -n 's/.*<!-- q10-verdict: \([a-z-]*\) -->.*/\1/p' docs/RESULTS.md)"
  [[ "${expected%% *}" == "$recorded" ]] || { echo ">> Q10's rule gives '${expected%% *}' ($expected), but RESULTS.md records '$recorded'"; return 1; }
  echo ">> Q10: $expected; recorded verdict agrees"
  local a=runs/p4/unet_sup.bin b=runs/q1/unet_levels60.bin threshold
  [[ -f "$FMD/test_mix/.complete" && -f $a && -f $b ]] \
    || { echo "verdict checked on the recorded results; reproduction skipped (needs FMD data and the P4/Q1 weights)"; return 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  reproduces q10-validation fmd-route --weights $a --weights-clean $b --root "$FMD" --seed 2026 || return 1
  threshold="$(recorded_block q10-validation | sed -n 's/^chosen noise threshold: //p')"
  reproduces q10-routed fmd-eval --weights $a --weights-clean $b --noise-threshold "$threshold" --root "$FMD" --levels raw,avg2,avg4,avg8,avg16 || return 1
}

gate_28() (   # cli-batch (subshell: set -e stops only this gate)
  set -e
  test_gate 28-cli-batch
  dotnet build tools/DoraBio.Cli -c Release >/dev/null
  local t="$LOG_DIR/batch" w=godot/models/dora_bio_weights.bin
  dotnet "$CLI_DLL" synth --out "$t/data" --count 3 --size 48 --seed 8 --stack "$t/stack.tif" >/dev/null
  dotnet "$CLI_DLL" denoise --weights $w --in "$t/stack.tif" --out "$t/stack_out.tif" --baseline "$t/stack_base.tif" | tee "$t/stack.log"
  grep -q "^✨ denoised 3 pages" "$t/stack.log"
  grep -q "^   peak memory: [0-9]* MB" "$t/stack.log"
  dotnet "$CLI_DLL" denoise --weights $w --in "$t/data/noisy" --out "$t/folder" | tee "$t/folder.log"
  grep -q "^✨ denoised 3 files, 3 pages" "$t/folder.log"
  test -s "$t/stack_base.tif" && test -s "$t/folder/synthetic_002.tif"
  # Peak memory must not grow with the stack's length: 40 pages of 512² may peak at most 160 MB above 3 pages (measured:
  # about 180 MB for 3 pages and +90 MB for 40; without the batch's periodic collection, +290 MB, and still rising).
  local p3 p40
  dotnet "$CLI_DLL" synth --out "$t/mem3" --count 3 --size 512 --seed 3 --stack "$t/mem3.tif" >/dev/null
  dotnet "$CLI_DLL" synth --out "$t/mem40" --count 40 --size 512 --seed 3 --stack "$t/mem40.tif" >/dev/null
  p3="$(dotnet "$CLI_DLL" denoise --weights $w --in "$t/mem3.tif" --out "$t/mem3_out.tif" | sed -n 's/^   peak memory: \([0-9]*\) MB$/\1/p')"
  p40="$(dotnet "$CLI_DLL" denoise --weights $w --in "$t/mem40.tif" --out "$t/mem40_out.tif" | sed -n 's/^   peak memory: \([0-9]*\) MB$/\1/p')"
  echo "peak memory: 3 pages $p3 MB, 40 pages $p40 MB"
  (( p40 - p3 <= 160 )) || { echo ">> peak memory grows with the stack's length (+$((p40 - p3)) MB for 37 more pages)"; exit 1; }
  # The stack's pages must equal the folder's files: Pillow, an independent reader, compares them when installed.
  python3 -I -c "import PIL" 2>/dev/null || { echo "page-for-file comparison skipped (Pillow not installed); unit tests cover it"; exit 0; }
  python3 -I - "$t/stack_out.tif" "$t/folder" <<'PY'
import sys
from pathlib import Path
from PIL import Image
stack = Image.open(sys.argv[1])
files = sorted(Path(sys.argv[2]).glob("*.tif"))
assert stack.n_frames == len(files) == 3, (stack.n_frames, len(files))
for page, file in enumerate(files):
    stack.seek(page)
    single = Image.open(file)
    assert stack.mode == single.mode == "F" and stack.tobytes() == single.tobytes(), f"page {page} differs from {file.name}"
print("stack pages equal the folder's files (Pillow)")
PY
)

gate_29() {   # q13-conditioning (quality pass Q13, pre-registered in PATHWAY.md, commit c14066f, before any of its code)
  test_gate 29-q13-conditioning || return 1
  # Rule (seed 2026, test_mix all 48): raw >= A's raw - 0.1 (A = p4-unet-sup); mean over avg2..avg16 >= B's mean - 0.1
  # (B = q1-candidate, the plain all-level U-Net, seed 2026); no level below B's - 0.2; and over seeds 2026-2028 the
  # conditioned recipe's mean raw >= the plain recipe's + 0.2. All met: adopted. Else raw >= B's raw + 0.2 and < A - 0.1:
  # improvement-recorded. Else: not-adopted.
  local block blocks=(p4-unet-sup q1-candidate q13-cond-2026 q13-cond-2027 q13-cond-2028 q13-plain-2027 q13-plain-2028)
  for block in "${blocks[@]}"; do
    [[ -n "$(recorded_block "$block")" ]] || { echo ">> docs/RESULTS.md needs the $block block"; return 1; }
  done
  local expected recorded
  expected="$( { echo "a raw $(recorded_block p4-unet-sup | awk '$1 == "all" { print $9 }')"
                 recorded_block q1-candidate | levels_all | sed 's/^/b /'
                 for block in "${blocks[@]:2}"; do recorded_block "$block" | levels_all | sed "s/^/${block#q13-} /"; done; } | awk '
    { v[$1 "," $2] = $3 }
    END {
      split("avg2 avg4 avg8 avg16", L, " "); cm = 0; bm = 0; low = 1
      for (i = 1; i <= 4; i++) {
        c = v["cond-2026," L[i]]; b = v["b," L[i]]; cm += c / 4; bm += b / 4
        if (c < b - 0.2 - 1e-9) low = 0
      }
      craw = v["cond-2026,raw"]
      seeds = (v["cond-2026,raw"] + v["cond-2027,raw"] + v["cond-2028,raw"]) / 3 - (v["b,raw"] + v["plain-2027,raw"] + v["plain-2028,raw"]) / 3
      met = craw >= v["a,raw"] - 0.1 - 1e-9 && cm >= bm - 0.1 - 1e-9 && low && seeds >= 0.2 - 1e-9
      verdict = met ? "adopted" : (craw >= v["b,raw"] + 0.2 - 1e-9 && craw < v["a,raw"] - 0.1 - 1e-9 ? "improvement-recorded" : "not-adopted")
      printf "%s raw %+.2f vs A, %+.2f vs B; lower levels %+.2f vs B; seed-mean raw gain over plain %+.2f\n", verdict, craw - v["a,raw"], craw - v["b,raw"], cm - bm, seeds
    }')"
  recorded="$(sed -n 's/.*<!-- q13-verdict: \([a-z-]*\) -->.*/\1/p' docs/RESULTS.md)"
  [[ "${expected%% *}" == "$recorded" ]] || { echo ">> Q13's rule gives '${expected%% *}' ($expected), but RESULTS.md records '$recorded'"; return 1; }
  echo ">> Q13: $expected; recorded verdict agrees"
  # The shipped default follows the verdict: conditioning is opt-in unless Q13 was adopted.
  local default_condition
  default_condition="$(dotnet "$CLI_DLL" info --default-condition supervised)" || return 1
  [[ ( "$recorded" == adopted && "$default_condition" == noise ) || ( "$recorded" != adopted && "$default_condition" == none ) ]] \
    || { echo ">> supervised training's default --condition is '$default_condition', but Q13 was '$recorded'"; return 1; }
  local seed name
  for name in cond_s2026 cond_s2027 cond_s2028 plain_s2027 plain_s2028; do
    [[ -f runs/q13/$name.bin ]] || { echo "verdict checked on the recorded results; reproduction skipped (needs FMD data and runs/q13 weights)"; return 77; }
  done
  [[ -f "$FMD/test_mix/.complete" ]] || { echo "verdict checked on the recorded results; reproduction skipped (needs FMD data)"; return 77; }
  for name in cond_s2026 cond_s2027 cond_s2028 plain_s2027 plain_s2028; do
    block="q13-${name/_s/-}"
    reproduces "$block" fmd-eval --weights runs/q13/$name.bin --root "$FMD" --levels raw,avg2,avg4,avg8,avg16 || return 1
  done
}

gate_30() {   # ext-data (field validation: the public paired datasets from tools/fetch_ext.sh)
  bash -n tools/fetch_ext.sh || return 1
  local entries pinned
  entries="$(grep -cE '^  "[a-z0-9-]+\|' tools/fetch_ext.sh)"
  pinned="$(grep -cE '^  "[a-z0-9-]+\|[0-9]+\|[^|]+\.zip\|[0-9]+\|[0-9a-f]{64}"$' tools/fetch_ext.sh)"
  [[ "$entries" -eq 4 && "$pinned" -eq 4 ]] || { echo ">> $pinned of $entries archives fully pinned (need 4 with record, file, size and sha256)"; return 1; }
  test_gate 30-ext-data || return 1
  local name sha missing=0
  while IFS='|' read -r name _ _ _ sha; do
    sha="${sha%\"}"
    if [[ ! -f "$EXT/$name/.complete" ]]; then missing=$((missing + 1)); continue; fi
    [[ "$(cat "$EXT/$name/.complete")" == "$sha" ]] || { echo ">> $name was extracted from an archive that doesn't match its pin"; return 1; }
  done < <(grep -E '^  "[a-z0-9-]+\|' tools/fetch_ext.sh | sed 's/^  "//')
  [[ $missing -lt 4 ]] || { echo "pins and unit tests OK; data check skipped (run tools/fetch_ext.sh)"; return 77; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  dotnet "$CLI_DLL" ext-check --root "$EXT" || return 1
  [[ $missing -eq 0 ]] || { echo "present datasets verified; $missing not downloaded"; return 77; }
}

gate_31() {   # q18-ext-bench (quality pass Q18, pre-registered in PATHWAY.md, commit be5228b, before any of its code)
  test_gate 31-q18-ext-bench || return 1
  # Rule: averaged over the three datasets, DORA supervised >= CAREamics CARE - 0.5 dB and DORA Noise2Void >= CAREamics
  # Noise2Void - 0.5 dB (mean test PSNR after CARE's normalization): competitive. Otherwise: not-competitive.
  local d m datasets=(mreb nucleoid care2d) expected recorded
  for d in "${datasets[@]}"; do for m in dora-n2v dora-sup careamics-n2v careamics-care; do
    [[ -n "$(recorded_block "q18-$m-$d")" ]] || { echo ">> docs/RESULTS.md needs the q18-$m-$d block"; return 1; }
  done; done
  expected="$( for d in "${datasets[@]}"; do for m in dora-n2v dora-sup careamics-n2v careamics-care; do
                 recorded_block "q18-$m-$d" | awk -v k="$m" '$1 == "mean" { print k, $8 }'; done; done | awk '
    { sum[$1] += $2 / 3 }
    END {
      n2v = sum["dora-n2v"] - sum["careamics-n2v"]; sup = sum["dora-sup"] - sum["careamics-care"]
      printf "%s Noise2Void %+.2f dB, supervised %+.2f dB vs CAREamics (mean over 3 datasets)\n", \
        (n2v >= -0.5 - 1e-9 && sup >= -0.5 - 1e-9) ? "competitive" : "not-competitive", n2v, sup
    }')"
  recorded="$(sed -n 's/.*<!-- q18-verdict: \([a-z-]*\) -->.*/\1/p' docs/RESULTS.md)"
  [[ "${expected%% *}" == "$recorded" ]] || { echo ">> Q18's rule gives '${expected%% *}' ($expected), but RESULTS.md records '$recorded'"; return 1; }
  echo ">> Q18: $expected; recorded verdict agrees"
  local -A full=([mreb]=deepbacs-mreb [nucleoid]=deepbacs-nucleoid [care2d]=zerocost-care2d)
  for d in "${datasets[@]}"; do
    [[ -f "$EXT/${full[$d]}/.complete" && -f runs/q18/dora_sup_${full[$d]}.bin && -f runs/q18/careamics_care/${full[$d]}/timing.json ]] \
      || { echo "verdict checked on the recorded results; reproduction skipped (needs data/ext and runs/q18)"; return 77; }
  done
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  for d in "${datasets[@]}"; do
    local name="${full[$d]}"
    reproduces "q18-dora-n2v-$d" ext-eval --dataset "$name" --root "$EXT" --weights "runs/q18/dora_n2v_$name.bin" || return 1
    reproduces "q18-dora-sup-$d" ext-eval --dataset "$name" --root "$EXT" --weights "runs/q18/dora_sup_$name.bin" || return 1
    reproduces "q18-careamics-n2v-$d" ext-eval --dataset "$name" --root "$EXT" --predictions "runs/q18/careamics_n2v/$name" \
      --label "CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU" || return 1
    reproduces "q18-careamics-care-$d" ext-eval --dataset "$name" --root "$EXT" --predictions "runs/q18/careamics_care/$name" \
      --label "CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU" || return 1
  done
}

gate_32() {   # quant-fidelity
  test_gate 32-quant-fidelity || return 1
  local n count=0 skipped=0
  count="$(grep -c '^<!-- fidelity-' docs/RESULTS.md)"
  [[ "$count" -eq 18 ]] || { echo ">> docs/RESULTS.md should hold 18 fidelity blocks, found $count"; return 1; }
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  local -A fmd=([bundled]=godot/models/dora_bio_weights.bin [sup-dilated]=runs/p3/sup_plain.bin [sup-unet]=runs/p4/unet_sup.bin
                [sup-unet-levels]=runs/q1/unet_levels60.bin [n2v-dilated]=runs/p3/n2v_plain.bin [n2v-unet]=runs/p4/unet_n2v.bin)
  for n in bundled sup-dilated sup-unet sup-unet-levels n2v-dilated n2v-unet; do
    if [[ -f "$FMD/test_mix/.complete" && -f "${fmd[$n]}" ]]; then
      reproduces "fidelity-fmd-$n" fidelity --fmd "$FMD" --weights "${fmd[$n]}" || return 1
    else skipped=$((skipped + 1)); fi
  done
  local d
  for d in deepbacs-mreb deepbacs-nucleoid zerocost-care2d; do
    if [[ -f "$EXT/$d/.complete" && -f "runs/q18/dora_sup_$d.bin" && -f "runs/q18/careamics_care/$d/timing.json" ]]; then
      reproduces "fidelity-$d-dora-n2v" fidelity --dataset "$d" --root "$EXT" --weights "runs/q18/dora_n2v_$d.bin" || return 1
      reproduces "fidelity-$d-dora-sup" fidelity --dataset "$d" --root "$EXT" --weights "runs/q18/dora_sup_$d.bin" || return 1
      reproduces "fidelity-$d-careamics-n2v" fidelity --dataset "$d" --root "$EXT" --predictions "runs/q18/careamics_n2v/$d" \
        --label "CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU" || return 1
      reproduces "fidelity-$d-careamics-care" fidelity --dataset "$d" --root "$EXT" --predictions "runs/q18/careamics_care/$d" \
        --label "CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU" || return 1
    else skipped=$((skipped + 4)); fi
  done
  [[ $skipped -eq 0 ]] || { echo "unit tests passed; $skipped of 18 fidelity blocks not re-run (their data or weights are missing)"; return 77; }
}

gate_21() {   # arch-eval
  test_gate 21-arch-eval || return 1
  # The rule was set before measuring (PATHWAY.md, P4): the U-Net becomes the default only if, at the same training
  # steps, it beats the dilated CNN on all-48 test_mix PSNR (raw) for supervised AND Noise2Void training, and is at
  # most 3x slower on a 512x512 frame. The recorded blocks hold the four scores; the default must follow the rule.
  local dncnn_sup unet_sup dncnn_n2v unet_n2v
  dncnn_sup="$(recorded_block p4-dncnn-sup | awk '$1 == "all" { print $9 }')"
  unet_sup="$(recorded_block p4-unet-sup | awk '$1 == "all" { print $9 }')"
  dncnn_n2v="$(recorded_block p4-dncnn-n2v | awk '$1 == "all" { print $9 }')"
  unet_n2v="$(recorded_block p4-unet-n2v | awk '$1 == "all" { print $9 }')"
  [[ -n "$dncnn_sup" && -n "$unet_sup" && -n "$dncnn_n2v" && -n "$unet_n2v" ]] \
    || { echo ">> docs/RESULTS.md needs p4-dncnn-sup, p4-unet-sup, p4-dncnn-n2v and p4-unet-n2v fmd-eval blocks"; return 1; }
  local slowdown; slowdown="$(sed -n 's/.*<!-- p4-slowdown: \([0-9.]*\) -->.*/\1/p' docs/RESULTS.md)"
  [[ -n "$slowdown" ]] || { echo ">> docs/RESULTS.md needs a <!-- p4-slowdown: X --> marker (U-Net / dilated CNN time, 512x512)"; return 1; }
  local want=dilated
  awk -v a="$unet_sup" -v b="$dncnn_sup" -v c="$unet_n2v" -v d="$dncnn_n2v" -v s="$slowdown" \
    'BEGIN { exit !(a > b && c > d && s <= 3) }' && want=unet
  dotnet build tools/DoraBio.Cli -c Release >/dev/null || return 1
  local have; have="$(dotnet "$CLI_DLL" info --default-arch)"
  [[ "$have" == "$want" ]] || { echo ">> the rule says the default should be $want, but NetConfig.Default is $have"; return 1; }
  echo ">> supervised: U-Net $unet_sup vs dilated $dncnn_sup dB; Noise2Void: U-Net $unet_n2v vs dilated $dncnn_n2v dB; slowdown ${slowdown}x: default $want"
  local w=runs/p4 d=runs/p3
  [[ -f "$FMD/test_mix/.complete" && -f $d/sup_plain.bin && -f $d/n2v_plain.bin && -f $w/unet_sup.bin && -f $w/unet_n2v.bin ]] \
    || { echo "rule checked on the recorded results; reproduction skipped (needs FMD data and the weights from tools/experiments)"; return 77; }
  reproduces p4-dncnn-sup fmd-eval --weights $d/sup_plain.bin --root "$FMD" --levels raw || return 1
  reproduces p4-unet-sup fmd-eval --weights $w/unet_sup.bin --root "$FMD" --levels raw || return 1
  reproduces p4-dncnn-n2v fmd-eval --weights $d/n2v_plain.bin --root "$FMD" --levels raw || return 1
  reproduces p4-unet-n2v fmd-eval --weights $w/unet_n2v.bin --root "$FMD" --levels raw || return 1
}

# recorded_block <marker>: the contents of the ```text block between <!-- marker --> and <!-- /marker --> in RESULTS.md.
recorded_block() { sed -n "/<!-- $1 -->/,/<!-- \/$1 -->/p" docs/RESULTS.md | sed '1,2d;$d' | sed '$d'; }

# reproduces <marker> <cli args...>: the recorded block is the verbatim output of `dora-bio <cli args...>`, and a
# fresh run must print exactly the same.
reproduces() {
  local marker="$1" recorded fresh; shift
  recorded="$(recorded_block "$marker")"
  [[ -n "$recorded" ]] || { echo ">> no <!-- $marker --> block in docs/RESULTS.md"; return 1; }
  fresh="$(dotnet "$CLI_DLL" "$@")" || return 1
  if [[ "$recorded" != "$fresh" ]]; then
    echo ">> RESULTS.md block '$marker' differs from a fresh run:"; diff <(echo "$recorded") <(echo "$fresh") | head -20; return 1
  fi
  echo ">> RESULTS.md block '$marker' reproduces exactly"
}

# ---- gate self-test ---------------------------------------------------------
# A gate that can't fail proves nothing. For each gate that has code of its own, plant one known bug in a fresh scratch copy of
# this folder and check that exactly that gate turns red. Mutations are exact string replacements that must
# match once. If the code moves on and a mutation no longer applies, the self-test fails and says so.
MUTATIONS=(
  # gate | file | text to find | replacement ("<DELETE-FILE>" deletes the file)
  $'0|tests/DoraBio.Core.Tests/DataTests.cs|[Trait(Bricks.Trait, Bricks.Data)]\n|'
  $'0|src/DoraBio.Core/Data/GrayImage.cs|    /// <summary>Width in pixels (number of columns).</summary>\n|'
  '1|src/DoraBio.Core/Nn/ConvLayer.Blocked.cs|gradients[BiasOffset + oc] += Simd.Sum(gradOutput.Slice(oc * plane, plane));|gradients[BiasOffset + oc] += 1.1f * Simd.Sum(gradOutput.Slice(oc * plane, plane));'
  '2|src/DoraBio.Core/Data/Dihedral.cs|(a, b) = (b, last - a);|(a, b) = (b, a);'
  '3|src/DoraBio.Core/IO/PngDecoder.cs|3 => (a + b) >> 1,|3 => (a + b + 1) >> 1,'
  '4|src/DoraBio.Core/TiledInference.cs|margin = RoundUp(net.Config.ReceptiveRadius, align);|margin = RoundUp(net.Config.ReceptiveRadius, align) - 1;'
  '5|src/DoraBio.Core/Training/Trainer.cs|old * cfg.LearningRateDecay|old / cfg.LearningRateDecay'
  '6|godot/samples/synthetic_cells.tif||<DELETE-FILE>'
  $'7|godot/scripts/DoraBioBridge.cs|    /// <summary>True while a denoise, adapt or training job runs. Only one job runs at a time.</summary>\n|'
  '11|tools/fetch_fmd.sh|d7a14d3be17368dbfe24a5cfe8ed0d40d403d7902d70fe914d04303e4aa8f2f6"|"'
  '12|src/DoraBio.Core/Data/FmdDataset.cs|public IReadOnlyCollection<int> ExcludedFovs { get; init; } = [FmdDataset.TestFov];|public IReadOnlyCollection<int> ExcludedFovs { get; init; } = [];'
  '13|docs/RESULTS.md||<DELETE-FILE>'
  '14|src/DoraBio.Core/NoiseDiagnostics.cs|b = r[(y + dy) * TileSize + x + dx];|b = r[(y + dy) * TileSize + x];'
  '15|src/DoraBio.Core/Data/BlindSpot.cs|do q = rng.Next(plane); while (hidden[q]);|q = rng.Next(plane);'
  '16|src/DoraBio.Core/Data/Anscombe.cs|- 11 / (8 * d * d)|- 11 / (8 * d)'
  '17|src/DoraBio.Core/NoiseCalibration.cs|double gain = sxx > 0 ? sxy / sxx : 0;|double gain = sxx > 0 ? 0.8 * sxy / sxx : 0;'
  '18|src/DoraBio.Core/Nn/Resample.cs|float v = 0.25f * g[x];|float v = 0.5f * g[x];'
  '19|src/DoraBio.Core/Nn/DenoisingNet.UNet.cs|Simd.Axpy(1f, ws.SkipGrad[l].AsSpan(0, gOut.Length), gOut);|Simd.Axpy(0f, ws.SkipGrad[l].AsSpan(0, gOut.Length), gOut);'
  $'20|src/DoraBio.Core/TiledInference.cs|        tileCore = RoundUp(tileCore, align);\n|'
  '21|src/DoraBio.Core/Nn/DenoisingNet.cs|public static NetConfig Default => new(24, [1, 2, 4, 8, 4, 2, 1]);|public static NetConfig Default => UNet();'
  '22|src/DoraBio.Core/PerformanceBench.cs|? "unet" : "dilated";|? "u-net" : "dilated";'
  '23|src/DoraBio.Core/Nn/ConvLayer.Blocked.cs|a7 += new Vector<float>(Unsafe.Add(ref weights, t + 7 * ic9)) * v;|a7 += new Vector<float>(Unsafe.Add(ref weights, t + 6 * ic9)) * v;'
  '24|src/DoraBio.Core/Nn/DenoisingNet.cs|    internal bool ReferenceKernels { get; set; }|    internal bool ReferenceKernels { get; set; } = true;'
  '25|docs/RESULTS.md|<!-- q1-verdict: not-adopted -->|<!-- q1-verdict: adopted -->'
  '26|src/DoraBio.Core/IO/TiffStack.cs|_pointerAt = ifdOffset + ifd.Length - 4;|_pointerAt = 4;'
  '27|docs/RESULTS.md|<!-- q10-verdict: not-adopted -->|<!-- q10-verdict: adopted -->'
  '28|src/DoraBio.Core/BatchDenoising.cs|output.Append(result.Output);|output.Append(raw);'
  '28|src/DoraBio.Core/BatchDenoising.cs|                GC.Collect();|'
  '29|src/DoraBio.Core/Nn/DenoisingNet.Film.cs|gradients[FilmW1 + i] += dHidden[i] * ws.Condition;|gradients[FilmW1 + i] += dHidden[i];'
  '29|docs/RESULTS.md|<!-- q13-verdict: not-adopted -->|<!-- q13-verdict: adopted -->'
  '30|src/DoraBio.Core/Data/ExtDatasets.cs|.Where(n => !clean.Contains(n)).ToList();|.Where(n => false).ToList();'
  '31|src/DoraBio.Core/ExtBenchmark.cs|double a = var > 0 ? cov / var : 0, b = mr - a * mx;|double a = 1, b = mr - a * mx;'
  '31|docs/RESULTS.md|<!-- q18-verdict: not-competitive -->|<!-- q18-verdict: competitive -->'
  '32|src/DoraBio.Core/Fidelity.cs|double logGain = mo - mr;|double logGain = 0;'
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
  printf "▶ brick %d %-15s " "$i" "${NAMES[$i]}"
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
