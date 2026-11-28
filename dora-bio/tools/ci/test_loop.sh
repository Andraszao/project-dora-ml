#!/usr/bin/env bash
# Hunts intermittent test failures (brick 41): runs the whole DoraBio.Core suite ROUNDS times in a scratch copy of the
# tree, cycling through runtime settings that change when locals stop being GC roots or how threads are scheduled,
# and keeps every TRX log, so a failure that shows up once leaves its test name, message and stack behind.
#   default | TieredCompilation=0 (optimized from the start) | TC_QuickJitForLoops=0 | TieredPGO=0 | gcServer=1 |
#   Debug build (locals live to the end of each method) | default with every core busy (contention)
# Each configuration builds fresh on its first run, and every third round rebuilds Release (the reported failure
# came right after a rebuild).
#   tools/ci/test_loop.sh ROUNDS OUT_DIR       (7 runs per round, about 3 minutes per round on 4 cores)
# Writes OUT_DIR/summary.txt (one line per run, failed tests named) and OUT_DIR/logs/run_N.trx. Exit 1 if any run failed.
set -u
rounds="${1:?usage: tools/ci/test_loop.sh ROUNDS OUT_DIR}"; out="${2:?usage: tools/ci/test_loop.sh ROUNDS OUT_DIR}"
src="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$out" && out="$(cd "$out" && pwd)"
work="$out/tree"; logs="$out/logs"
rm -rf "$work" "$logs" && mkdir -p "$work" "$logs"
(cd "$src" && tar --exclude=bin --exclude=obj --exclude=.godot --exclude=./data --exclude=./runs -cf - .) | tar -xf - -C "$work"
cd "$work"
: > "$out/summary.txt"
configs=(default tc0 qjfl0 pgo0 gcserver debug contention)
built_release=0; built_debug=0; i=0; failures=0
for r in $(seq 1 "$rounds"); do
  for c in "${configs[@]}"; do
    i=$((i + 1))
    cfg=Release; envs=()
    case $c in
      tc0) envs=(DOTNET_TieredCompilation=0) ;;
      qjfl0) envs=(DOTNET_TC_QuickJitForLoops=0) ;;
      pgo0) envs=(DOTNET_TieredPGO=0) ;;
      gcserver) envs=(DOTNET_gcServer=1) ;;
      debug) cfg=Debug ;;
    esac
    how="reused build"
    if [[ $cfg == Release && ( $built_release -eq 0 || ( $c == default && $((r % 3)) -eq 0 ) ) ]] || [[ $cfg == Debug && $built_debug -eq 0 ]]; then
      rm -rf tests/DoraBio.Core.Tests/{bin,obj}/$cfg src/DoraBio.Core/{bin,obj}/$cfg
      dotnet build tests/DoraBio.Core.Tests -c $cfg >"$logs/build_$i.log" 2>&1 || { echo "run $i ($c): build failed" >> "$out/summary.txt"; failures=$((failures + 1)); continue; }
      if [[ $cfg == Release ]]; then built_release=1; else built_debug=1; fi
      how="fresh build"
    fi
    stress=()
    if [[ $c == contention ]]; then
      for _ in $(seq 1 "$(nproc)"); do (timeout 300 sh -c 'while :; do :; done') & stress+=($!); done
    fi
    start=$(date +%s)
    env "${envs[@]}" dotnet test tests/DoraBio.Core.Tests -c $cfg --no-build --logger "trx;LogFileName=run_$i.trx" \
      --results-directory "$logs" >"$logs/run_$i.log" 2>&1
    code=$?
    secs=$(( $(date +%s) - start ))
    for p in "${stress[@]}"; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
    failed="$(grep -o 'testName="[^"]*" [^>]*outcome="Failed"' "$logs/run_$i.trx" 2>/dev/null | sed 's/testName="\([^"]*\)".*/\1/' | tr '\n' ';')"
    totals="$(grep -o '<Counters [^>]*' "$logs/run_$i.trx" 2>/dev/null | grep -o 'total="[0-9]*"\|passed="[0-9]*"\|failed="[0-9]*"' | tr '\n' ' ')"
    [[ $code -eq 0 ]] || failures=$((failures + 1))
    echo "run $i ($c, $cfg, $how, ${secs}s, exit $code): $totals${failed:+FAILED: $failed}" >> "$out/summary.txt"
  done
done
echo "done: $i runs, $failures failed" >> "$out/summary.txt"
cat "$out/summary.txt"
[[ $failures -eq 0 ]]
