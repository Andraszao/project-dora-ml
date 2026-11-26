#!/usr/bin/env bash
# Downloads the public paired denoising datasets of the field-validation track (PATHWAY.md, FV) into data/ext/
# (gitignored). Each comes from Zenodo under CC BY 4.0: credit the authors (see data/ext/README.txt) and never commit
# the data to this repo.
#
#   tools/fetch_ext.sh                  every dataset below (about 740 MB)
#   tools/fetch_ext.sh deepbacs-mreb    one dataset by name
#
# Each archive's size and SHA-256 must match the pins below (Zenodo publishes MD5s; the SHA-256s were pinned on
# 2026-11-26 from downloads whose MD5s matched Zenodo's). Archives are checked entry by entry before extraction
# (tools/extract_zip.py: no absolute paths, "..", links or special files). Finished datasets are marked .complete and
# skipped on re-runs. An archive already in data/ext/_zips/ is verified and reused instead of downloaded again.
# Needs network access to zenodo.org.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/ext

# name | Zenodo record | file in the record | size in bytes | sha256
ARCHIVES=(
  "deepbacs-mreb|6460867|DeepBacs_Data_Denoising_E.coli_MreB.zip|118515594|c47f797c250edfc41e17f513379842a4e4e08931c829c01b144bf4b1120e1ab9"
  "deepbacs-nucleoid|5551112|DeepBacs_Data_Denoising_E.coli_H-NS-mScarlet-I_dataset.zip|8066921|267ae8eba9c62dd740e4a5b14d3334f8e29697e60b8657ddeb1dff2b69b4bc2b"
  "deepbacs-bsubtilis|5551135|DeepBacs_Data_Denoising_B.subtilis_FtsZ-GFP_dataset.zip|435654091|5e88c64491df80165f353cf0da59bfe2d22784800a90829ac2172b881052c9c3"
  "zerocost-care2d|3713330|CARE (2D).zip|178823351|0c16e3388e636a480172b798ffc512f96c6787e3c3f3814339c2fddaa23a3e08"
)

mkdir -p "$DEST/_zips"
cat > "$DEST/README.txt" <<'TXT'
Public denoising datasets downloaded by tools/fetch_ext.sh. All are CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/): credit the authors when you use them. Do not commit this folder.

deepbacs-mreb       Zenodo 10.5281/zenodo.6460867  Spahn & Heilemann: E. coli MreB-sfGFP, paired low/high SNR (16-bit)
deepbacs-nucleoid   Zenodo 10.5281/zenodo.5551112  Spahn & Heilemann: E. coli H-NS-mScarlet-I, paired low/high SNR (8-bit)
deepbacs-bsubtilis  Zenodo 10.5281/zenodo.5551135  Holden & Conduit: B. subtilis FtsZ-GFP time series, no ground truth (16-bit)
  These three are part of DeepBacs: Spahn et al., "DeepBacs for multi-task bacterial image analysis using open-source
  deep learning approaches", Communications Biology 5, 688 (2022).
zerocost-care2d     Zenodo 10.5281/zenodo.3713330  Jacquemet: CARE (2D) example training and test data (float32), from
  ZeroCostDL4Mic: von Chamier et al., "Democratising deep learning for microscopy with ZeroCostDL4Mic",
  Nature Communications 12, 2276 (2021).
TXT

fetched=0
for entry in "${ARCHIVES[@]}"; do
  IFS='|' read -r name record file size sha <<<"$entry"
  if [[ $# -gt 0 ]] && ! printf '%s\n' "$@" | grep -qxF "$name"; then continue; fi
  if [[ -f "$DEST/$name/.complete" ]]; then echo "✓ $name (already here)"; continue; fi

  zip="$DEST/_zips/$name.zip"
  if [[ ! -f "$zip" ]]; then
    echo "↓ $name ($((size / 1048576)) MiB, Zenodo record $record)"
    curl -fL --retry 3 --retry-delay 5 -sS -o "$zip.part" \
      "https://zenodo.org/records/$record/files/$(python3 -I -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$file")?download=1"
    mv "$zip.part" "$zip"
  fi
  actual="$(stat -c %s "$zip")"
  [[ "$actual" == "$size" ]] || { echo "  ✗ $name: size $actual, expected $size"; rm -f "$zip"; exit 1; }
  digest="$(sha256sum "$zip" | cut -d' ' -f1)"
  [[ "$digest" == "$sha" ]] || { echo "  ✗ $name: sha256 $digest does not match the pinned $sha"; rm -f "$zip"; exit 1; }

  rm -rf "${DEST:?}/$name"
  python3 -I tools/extract_zip.py "$zip" "$DEST/$name"
  echo "$digest" > "$DEST/$name/.complete"
  rm -f "$zip"
  echo "  ✓ $name: $(find "$DEST/$name" -type f -name '*.tif' | wc -l) TIFF files, sha256 ${digest:0:16}… (pinned, verified)"
  fetched=$((fetched + 1))
done
rmdir "$DEST/_zips" 2>/dev/null || true
echo "done: $fetched dataset(s) fetched into $DEST"
