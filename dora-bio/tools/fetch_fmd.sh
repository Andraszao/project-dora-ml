#!/usr/bin/env bash
# Downloads the Fluorescence Microscopy Denoising (FMD) dataset into data/fmd/ (gitignored).
#
#   tools/fetch_fmd.sh                 everything: 12 categories + the test_mix benchmark set (about 9 GB)
#   tools/fetch_fmd.sh test            just test_mix (43 MB)
#   tools/fetch_fmd.sh confocal        Confocal_* (also twophoton, widefield), or name categories directly
#
# Source: the authors' public Google Drive (the same files their own download_dataset.sh fetches), from
# Zhang et al., "A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images", CVPR 2019.
# License: CC BY-SA 4.0 (Notre Dame CurateND record). Attribution is required, and adaptations of the data
# must be shared alike. Never commit the data to this repo.
#
# Each archive is checked before extraction: its size must equal the size Google Drive reports for it
# (pinned below), its SHA-256 must match the value pinned after the first verified download, and it may
# contain only regular files and folders, with no absolute paths or "..". Finished categories are marked
# with .complete and skipped on re-runs, so an interrupted fetch can just be run again.
# Needs network access to drive.usercontent.google.com.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/fmd

# name | Google Drive file id | size in bytes (Drive metadata) | sha256, pinned on 2026-11-25 from downloads whose
# sizes matched Drive's metadata (trust on first use: the authors publish no checksums)
ARCHIVES=(
  "test_mix|13bAS6Ipuk2xZkF99FTdanJ3fenvA3QDj|42956800|d7a14d3be17368dbfe24a5cfe8ed0d40d403d7902d70fe914d04303e4aa8f2f6"
  "Confocal_BPAE_B|1juaumcGn5QlFRXRQyrqfbZBhF7oX__iW|589619200|4b0960757cc0ca7c0770c141214d60229a19d02cccd1df0d4d3b69b24b3aab2b"
  "Confocal_BPAE_G|1Zofz11VmI1JfRIMF7rq40RVjpzM6A9vg|790200320|d721e7618fc8ecd14d88fc5894c2f60e6e7c2fcb6d35f04660b3c4d690bbc955"
  "Confocal_BPAE_R|1QoD_vMvFdFg7yREfen3t-SGLFcnLg9YQ|627537920|9b36bb4df24ae81947d6829b1e7ae33c31eb03430614f0de7a3382034f3f81d2"
  "Confocal_FISH|1SxmsythWfxnfKJfGWpT_7Adebi8jUK98|697856000|16446776abb9c1e19d126a9300414a7e69afe70a862991442fb71164bc4ef289"
  "Confocal_MICE|11aflcrcatFRkv7EabjWjdlpT0DYRbUDZ|737976320|9da615b89b2670443ecf3e3d158e21c141364a0f893360f462d191ffb08a93af"
  "TwoPhoton_BPAE_B|1yVD_H_ZfNNSma5vtHZM_DTnSv1Bo1tfk|825835520|06cd148b90c6a8534da9b411624b2a8dfcc7647466ff8695572e2c8b6f689a18"
  "TwoPhoton_BPAE_G|125nqTfQQG1-YVUs256b2vTwt4aUNCgBt|873267200|b163128f043395547b3db0d1b69d52ec6031fad4f0e11795285ec9029fda9bac"
  "TwoPhoton_BPAE_R|1rwxG6LYcKeiBKNT3Oq9lvwKu8mV3rz9P|696156160|9315a57d606116fb506bcaf07af81096f149ba7f51827762893bfb290e2eed08"
  "TwoPhoton_MICE|1lhsFAlXsXk26yqHzT0_-3R8MUb7G0NVa|870973440|4d709fe0c3a5b70da62796ec9029809cbbdc5c1d5f49dae0aad940ab653d8e5f"
  "WideField_BPAE_B|19rl8zFzfXIZ2drgodCGutLPLzL4kJq6d|718602240|a4db37c7094526848b1667ea159481278a5f00254fed64f65d5b83cf921537d6"
  "WideField_BPAE_G|1H67O6GqIkIlQSX-n0vfMWGPwmd4zOHQr|792780800|019863658ab0ba45c2b323ef787bad2ed40b017c0cdcfae66eb7818eaf9bdeee"
  "WideField_BPAE_R|19HXb2Ftrb-M7Lr9ZlHWMcnNT0Sbu85YL|709232640|4914cd7d951b4ddc1a01f6c7f121b7e9936fd2a7d1505f3e802984ffee69cad7"
)

wanted() {   # wanted <name> <selectors...>
  local name="$1"; shift
  [[ $# -eq 0 ]] && return 0
  for sel in "$@"; do
    case "$sel" in
      all) return 0 ;;
      test) [[ "$name" == test_mix ]] && return 0 ;;
      confocal) [[ "$name" == Confocal_* ]] && return 0 ;;
      twophoton) [[ "$name" == TwoPhoton_* ]] && return 0 ;;
      widefield) [[ "$name" == WideField_* ]] && return 0 ;;
      *) [[ "$name" == "$sel" ]] && return 0 ;;
    esac
  done
  return 1
}

# Refuses archives with anything but regular files and folders, absolute paths, or ".." components.
check_archive() {
  local tar="$1" bad
  bad="$(tar -tvf "$tar" | awk '$1 !~ /^[-d]/' | head -3)"
  [[ -z "$bad" ]] || { echo "  ✗ $tar has non-regular entries (links/devices):"; echo "$bad"; return 1; }
  bad="$(tar -tf "$tar" | grep -E '^/|(^|/)\.\.(/|$)' | head -3 || true)"
  [[ -z "$bad" ]] || { echo "  ✗ $tar has unsafe paths:"; echo "$bad"; return 1; }
}

mkdir -p "$DEST/_tars"
cat > "$DEST/README.txt" <<'TXT'
Fluorescence Microscopy Denoising (FMD) dataset, downloaded by tools/fetch_fmd.sh.
Zhang, Zhu, Nichols, Wang, Zhang, Smith, Howard: "A Poisson-Gaussian Denoising Dataset with Real
Fluorescence Microscopy Images", CVPR 2019. arXiv:1812.10366.
License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Credit the authors; share
adaptations of the data under the same license. Do not commit this folder to the DORA repository.
TXT

fetched=0
for entry in "${ARCHIVES[@]}"; do
  IFS='|' read -r name id size sha <<<"$entry"
  wanted "$name" "$@" || continue
  if [[ -f "$DEST/$name/.complete" ]]; then echo "✓ $name (already here)"; continue; fi

  tar="$DEST/_tars/$name.tar"
  echo "↓ $name ($((size / 1048576)) MiB)"
  curl -fL --retry 3 --retry-delay 5 -sS -o "$tar.part" \
    "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t"
  actual="$(stat -c %s "$tar.part")"
  if [[ "$actual" != "$size" ]]; then
    echo "  ✗ size $actual, expected $size (Drive may have served an HTML page instead of the file)"; rm -f "$tar.part"; exit 1
  fi
  digest="$(sha256sum "$tar.part" | cut -d' ' -f1)"
  if [[ -n "$sha" && "$digest" != "$sha" ]]; then
    echo "  ✗ sha256 $digest does not match the pinned $sha"; rm -f "$tar.part"; exit 1
  fi
  mv "$tar.part" "$tar"
  check_archive "$tar"

  rm -rf "${DEST:?}/$name"
  tar -xf "$tar" -C "$DEST" --exclude='desktop.ini' --no-same-owner --no-same-permissions
  [[ -d "$DEST/$name" ]] || { echo "  ✗ archive did not contain $name/"; exit 1; }
  echo "$digest" > "$DEST/$name/.complete"
  rm -f "$tar"
  echo "  ✓ $name: $(find "$DEST/$name" -type f -name '*.png' | wc -l) images, sha256 ${digest:0:16}…${sha:+ (pinned, verified)}"
  fetched=$((fetched + 1))
done
rmdir "$DEST/_tars" 2>/dev/null || true
echo "done: $fetched archive(s) fetched into $DEST"
