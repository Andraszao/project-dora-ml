#!/usr/bin/env bash
# Publishes the dora-bio CLI as one self-contained executable (no .NET install needed to run it) and runs that
# executable end to end on synthetic data: `info` on the bundled model, `synth`, then `denoise --report` (brick 41).
# CI runs it on Linux, Windows and macOS runners (job publish-smoke); locally it checks this machine's build.
#   tools/ci/publish_smoke.sh RID        e.g. linux-x64, win-x64, osx-arm64, osx-x64
# Output goes to bin/publish-smoke/RID (gitignored, as every bin/ is). Exit 0 if every step works.
set -euo pipefail
rid="${1:?usage: tools/ci/publish_smoke.sh RID}"
cd "$(dirname "$0")/../.."
out="bin/publish-smoke/$rid"
rm -rf "$out"
dotnet publish tools/DoraBio.Cli -c Release -r "$rid" --self-contained true -p:PublishSingleFile=true \
  -p:IncludeNativeLibrariesForSelfExtract=true -p:DebugType=none -p:PublishDocumentationFile=false \
  -p:PublishReferencesDocumentationFiles=false -o "$out/app"
exe="$out/app/dora-bio"
[[ "$rid" == win-* ]] && exe="$exe.exe"
[[ -f "$exe" ]] || { echo ">> no executable at $exe:"; ls "$out/app"; exit 1; }
# One file: the executable and nothing else (no loose .dll a user could lose when copying it to another machine).
extra="$(ls "$out/app" | grep -vxF "$(basename "$exe")" || true)"
[[ -z "$extra" ]] || { echo ">> a single-file build should be one file, but $out/app also holds: $extra"; exit 1; }
echo "single file: $exe ($(( $(wc -c < "$exe") / 1048576 )) MB)"

# tr: the Windows build ends its lines with \r\n, which would break the anchored patterns below.
"$exe" info --weights godot/models/dora_bio_weights.bin | tr -d '\r' | tee "$out/info.txt"
grep -qE '^content hash *: sha256:[0-9a-f]{64}$' "$out/info.txt" || { echo ">> info should print the model's content hash"; exit 1; }
"$exe" synth --out "$out/synth" --count 1 --size 96 --seed 3
"$exe" denoise --weights godot/models/dora_bio_weights.bin --in "$out/synth/noisy/synthetic_000.tif" \
  --out "$out/denoised.tif" --report "$out/report.html"
[[ -s "$out/denoised.tif" ]] || { echo ">> denoise wrote no output"; exit 1; }
grep -q 'src="data:image/png;base64,' "$out/report.html" || { echo ">> the report should embed its picture"; exit 1; }
grep -qF "$(sed -n 's/^content hash *: //p' "$out/info.txt")" "$out/report.html" || { echo ">> the report should name the model's hash"; exit 1; }
echo ">> the single-file $rid build runs: info, synth, denoise --report"
