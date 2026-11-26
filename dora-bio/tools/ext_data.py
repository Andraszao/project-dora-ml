"""External datasets: the manifest's checks, the fetcher and ATTRIBUTIONS.md (standard library only; run with python3 -I).

    python3 -I tools/ext_data.py check                 validate tools/ext_manifest.json (schema, pins, licence policy)
    python3 -I tools/ext_data.py fetch [--full] [NAME]  download, verify and unpack into data/ext/ (tools/fetch_ext.sh)
    python3 -I tools/ext_data.py verify-local [NAME]   every downloaded dataset still matches its pins
    python3 -I tools/ext_data.py attributions [--check] write (or compare) ATTRIBUTIONS.md from the manifest

Without names, fetch takes the "core" tier (small, used by the default gates); --full adds the "full" tier (tens of GB).
Each file's size and MD5 must equal Zenodo's, and its SHA-256 the pinned value when one is pinned. A file not pinned
yet is accepted on its MD5, and its SHA-256 is printed so it can be pinned. Zips are checked entry by entry before
extraction (tools/extract_zip.py). Downloaded data is untrusted: it is only hashed, unpacked and read as images, never
executed. A finished dataset is marked data/ext/NAME/.complete (one SHA-256 per file, in manifest order).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "ext_manifest.json")
DEST = os.path.join(ROOT, "data", "ext")
ATTRIBUTIONS = os.path.join(ROOT, "ATTRIBUTIONS.md")

LICENCES = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}
TRAINABLE = {"CC-BY-4.0", "CC0-1.0"}   # owner decision 2026-11-26: a shipped model trains only on these (and synthetic data)
REQUIRED = ("name", "tier", "licence", "use", "doi", "creators", "title", "citation", "content", "files")


def load():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def problems(manifest):
    """Every way the manifest breaks its own rules (empty when it is sound)."""
    out, names = [], set()
    for d in manifest.get("datasets", []):
        name = d.get("name", "?")
        for key in REQUIRED:
            if key not in d:
                out.append(f"{name}: missing '{key}'")
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name):
            out.append(f"{name}: names are lowercase words joined by '-'")
        if name in names:
            out.append(f"{name}: listed twice")
        names.add(name)
        if d.get("licence") not in LICENCES:
            out.append(f"{name}: unknown licence {d.get('licence')!r} (known: {', '.join(LICENCES)})")
        if d.get("use") not in ("train", "eval-only"):
            out.append(f"{name}: use must be 'train' or 'eval-only'")
        if d.get("use") == "train" and d.get("licence") not in TRAINABLE:
            out.append(f"{name}: {d.get('licence')} data may only be used for evaluation (use: eval-only)")
        separate = "fetched_by" in d
        if d.get("tier") not in (("separate",) if separate else ("core", "full")):
            out.append(f"{name}: tier must be core or full (or 'separate' with fetched_by)")
        if separate and d.get("files"):
            out.append(f"{name}: a dataset fetched by {d['fetched_by']} lists no files here")
        if not separate:
            if d.get("doi") is None or not re.fullmatch(r"10\.5281/zenodo\.[0-9]+", d.get("doi", "")):
                out.append(f"{name}: needs its Zenodo DOI (10.5281/zenodo.N)")
            if not d.get("files"):
                out.append(f"{name}: lists no files")
        seen = set()
        for f in d.get("files", []):
            label = f"{name}/{f.get('file')}"
            if not f.get("file") or "/" in f["file"] or f["file"] in seen or f["file"].startswith("."):
                out.append(f"{label}: file names are plain, unique and not hidden")
            seen.add(f.get("file"))
            if not isinstance(f.get("size"), int) or f["size"] <= 0:
                out.append(f"{label}: size must be a positive byte count")
            if not re.fullmatch(r"[0-9a-f]{32}", f.get("md5") or ""):
                out.append(f"{label}: needs Zenodo's MD5 (32 hex digits)")
            if f.get("sha256") is not None and not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
                out.append(f"{label}: sha256 is 64 hex digits or null")
            if f.get("unpack") not in ("zip", "file"):
                out.append(f"{label}: unpack must be 'zip' or 'file'")
            if f.get("unpack") == "file" and not f.get("file", "").lower().endswith((".tif", ".tiff", ".mrc")):
                out.append(f"{label}: only images are placed unpacked")
    return out


def record_of(d):
    return d["doi"].rsplit(".", 1)[1]


def selected(manifest, names, full):
    known = {d["name"]: d for d in manifest["datasets"]}
    for n in names:
        if n not in known:
            sys.exit(f"unknown dataset '{n}'. Known: {', '.join(known)}")
    if names:
        return [known[n] for n in names]
    return [d for d in manifest["datasets"] if d["tier"] == "core" or (full and d["tier"] == "full")]


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def download(d, f, part):
    url = f"https://zenodo.org/records/{record_of(d)}/files/{urllib.parse.quote(f['file'])}?download=1"
    if os.path.exists(part) and os.path.getsize(part) == f["size"]:
        return
    if os.path.exists(part) and os.path.getsize(part) > f["size"]:
        os.remove(part)
    for attempt in range(1, 5):
        r = subprocess.run(["curl", "-fL", "--retry", "3", "--retry-delay", "5", "-sS", "-C", "-", "-o", part, url])
        if r.returncode == 0 and os.path.getsize(part) == f["size"]:
            return
        print(f"  … attempt {attempt} stopped at {os.path.getsize(part) if os.path.exists(part) else 0} bytes; resuming")
    sys.exit(f"  ✗ {d['name']}: could not download {f['file']}")


def fetch(names, full):
    manifest = load()
    bad = problems(manifest)
    if bad:
        sys.exit("manifest problems:\n  " + "\n  ".join(bad))
    os.makedirs(DEST, exist_ok=True)
    fetched = 0
    for d in selected(manifest, names, full):
        name, target = d["name"], os.path.join(DEST, d["name"])
        if "fetched_by" in d:
            print(f"· {name}: fetched by {d['fetched_by']}")
            continue
        if os.path.exists(os.path.join(target, ".complete")):
            print(f"✓ {name} (already here)")
            continue
        staging = os.path.join(DEST, "_downloads", name)
        os.makedirs(staging, exist_ok=True)
        total = sum(f["size"] for f in d["files"])
        print(f"↓ {name} ({total / 2**30:.2f} GiB, Zenodo record {record_of(d)})")
        digests = []
        for f in d["files"]:
            part = os.path.join(staging, f["file"])
            download(d, f, part)
            md5, sha = hashes(part)
            if md5 != f["md5"]:
                os.remove(part)
                sys.exit(f"  ✗ {name}/{f['file']}: MD5 {md5} is not Zenodo's {f['md5']} (deleted)")
            if f["sha256"] is not None and sha != f["sha256"]:
                os.remove(part)
                sys.exit(f"  ✗ {name}/{f['file']}: SHA-256 {sha} does not match the pinned {f['sha256']} (deleted)")
            if f["sha256"] is None:
                print(f"  ! {f['file']}: MD5 matches Zenodo; pin its SHA-256 in tools/ext_manifest.json: {sha}")
            digests.append(sha)
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target)
        for f in d["files"]:
            part = os.path.join(staging, f["file"])
            if f["unpack"] == "zip":
                r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "extract_zip.py"), part, target])
                if r.returncode != 0:
                    sys.exit(f"  ✗ {name}: {f['file']} failed the safety check; nothing kept")
                os.remove(part)
            else:
                os.replace(part, os.path.join(target, f["file"]))
        shutil.rmtree(os.path.join(target, "__MACOSX"), ignore_errors=True)   # macOS resource forks, not images
        with open(os.path.join(target, ".complete"), "w") as out:
            out.write("\n".join(digests) + "\n")
        shutil.rmtree(staging)
        images = sum(1 for _, _, fs in os.walk(target) for x in fs if x.lower().endswith((".tif", ".tiff", ".mrc")))
        print(f"  ✓ {name}: {images} image files, verified against Zenodo's MD5s{' and the pinned SHA-256s' if all(f['sha256'] for f in d['files']) else ''}")
        fetched += 1
    try:
        os.rmdir(os.path.join(DEST, "_downloads"))
    except OSError:
        pass
    write_readme(manifest)
    print(f"done: {fetched} dataset(s) fetched into data/ext")


def write_readme(manifest):
    lines = ["Public datasets downloaded by tools/fetch_ext.sh. Credit the authors (see ATTRIBUTIONS.md in the DORA",
             "repository for the full list and licences). Never commit this folder.", ""]
    for d in manifest["datasets"]:
        if "fetched_by" not in d:
            lines.append(f"{d['name']:<27} {d['licence']:<11} doi:{d['doi']}  {d['title']}")
    with open(os.path.join(DEST, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def verify_local(names):
    """Each dataset present under data/ext must have been unpacked from exactly its pinned files."""
    manifest = load()
    present = missing = 0
    failures = []
    for d in manifest["datasets"]:
        if "fetched_by" in d or (names and d["name"] not in names):
            continue
        marker = os.path.join(DEST, d["name"], ".complete")
        if not os.path.exists(marker):
            missing += 1
            continue
        present += 1
        with open(marker) as f:
            recorded = f.read().split()
        pins = [f["sha256"] for f in d["files"]]
        if None in pins:
            failures.append(f"{d['name']} is downloaded but not fully pinned: add these SHA-256s to the manifest: {' '.join(recorded)}")
        elif recorded != pins:
            failures.append(f"{d['name']} was unpacked from files that don't match its pins (re-run tools/fetch_ext.sh {d['name']})")
    for msg in failures:
        print(f">> {msg}")
    print(f"present {present} missing {missing}")
    return 1 if failures else 0


def attributions(manifest):
    """ATTRIBUTIONS.md, generated: the datasets DORA may download, with licence, permitted use and credit."""
    out = [
        "# Data attributions",
        "",
        "<!-- Generated by tools/ext_data.py attributions from tools/ext_manifest.json. Do not edit by hand. -->",
        "",
        "DORA downloads these public datasets for training and benchmarking. None of them is in this repository.",
        "**A model DORA ships is trained only on synthetic data and on datasets marked *train* below** (licences that",
        "allow it without share-alike terms: CC BY 4.0 or CC0). Datasets marked *eval-only* are benchmarks only. When",
        "you use any of them, credit the authors as listed.",
        "",
        "| Dataset | Licence | Use | Source | Credit |",
        "|---|---|---|---|---|",
    ]
    for d in manifest["datasets"]:
        source = f"[doi:{d['doi']}](https://doi.org/{d['doi']})" if d.get("doi") else f"`{d['fetched_by']}`"
        out.append(f"| `{d['name']}` | [{d['licence']}]({LICENCES[d['licence']]}) | {d['use']} | {source} | {d['creators']} |")
    out += ["", "## Citations", ""]
    for d in manifest["datasets"]:
        content = d["content"][0].upper() + d["content"][1:]
        out.append(f"- **`{d['name']}`**: {d['title']}. {content}. Cite: {d['citation']}.")
    out += ["", "## Considered and not used", ""]
    for x in manifest.get("excluded", []):
        doi = f" (doi:{x['doi']})" if x.get("doi") else ""
        out.append(f"- **{x['name']}**{doi}: {x['reason']}.")
    return "\n".join(out) + "\n"


def main(argv):
    sys.stdout.reconfigure(line_buffering=True)   # progress shows up live in logs
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        manifest = load()
        bad = problems(manifest)
        for msg in bad:
            print(f">> {msg}")
        ds = [d for d in manifest["datasets"] if "fetched_by" not in d]
        files = [f for d in ds for f in d["files"]]
        size = lambda tier: sum(f["size"] for d in ds if d["tier"] == tier for f in d["files"]) / 2**30
        print(f"{len(ds)} datasets ({sum(d['tier'] == 'core' for d in ds)} core, {sum(d['tier'] == 'full' for d in ds)} full), "
              f"{len(files)} files: core {size('core'):.2f} GiB, full {size('full'):.1f} GiB; "
              f"{sum(f['sha256'] is not None for f in files)} of {len(files)} SHA-256s pinned; "
              f"{sum(d['use'] == 'train' for d in manifest['datasets'])} trainable, "
              f"{sum(d['use'] == 'eval-only' for d in manifest['datasets'])} eval-only")
        return 1 if bad else 0
    if cmd == "fetch":
        fetch([a for a in rest if a != "--full"], "--full" in rest)
        return 0
    if cmd == "verify-local":
        return verify_local(rest)
    if cmd == "attributions":
        text = attributions(load())
        if "--check" in rest:
            current = open(ATTRIBUTIONS, encoding="utf-8").read() if os.path.exists(ATTRIBUTIONS) else ""
            if current != text:
                print(">> ATTRIBUTIONS.md is out of date: run python3 -I tools/ext_data.py attributions")
                return 1
            print("ATTRIBUTIONS.md matches the manifest")
            return 0
        with open(ATTRIBUTIONS, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(ATTRIBUTIONS, ROOT)}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
