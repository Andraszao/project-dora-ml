#!/usr/bin/env bash
# Downloads the public datasets of tools/ext_manifest.json into data/ext/ (gitignored): the field-validation sets
# (DeepBacs, ZeroCostDL4Mic) and the benchmark data of PATHWAY.md's road to state of the art (AI4Life challenges,
# BioSR+, DeepCAD-RT). Licences and credit: ATTRIBUTIONS.md. Never commit the data to this repo.
#
#   tools/fetch_ext.sh                    the "core" tier (about 0.9 GB; what the default gates use)
#   tools/fetch_ext.sh --full             core plus the "full" tier (about 60 GB)
#   tools/fetch_ext.sh mdc25-tribolium    datasets by name, from either tier
#
# Every file's size and MD5 must equal Zenodo's, and its SHA-256 the pinned value (pinned on 2026-11-26 from
# downloads whose MD5 matched). Zips are checked entry by entry before extraction (tools/extract_zip.py: no absolute
# paths, "..", links or special files). Interrupted downloads resume; finished datasets are marked .complete and
# skipped on re-runs. Needs network access to zenodo.org. The logic lives in tools/ext_data.py.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -I tools/ext_data.py fetch "$@"
