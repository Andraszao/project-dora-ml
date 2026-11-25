#!/usr/bin/env python3
"""Structural checks on .github/workflows/dora-bio.yml, used by bricks.sh gates 9 and 10.

    python3 tools/ci/check_workflow.py ../.github/workflows/dora-bio.yml core    # gate 9
    python3 tools/ci/check_workflow.py ../.github/workflows/dora-bio.yml godot   # gate 10

These are not a full GitHub Actions linter. The real proof is a green run on GitHub. They pin down the
properties DORA relies on:
  core:  follows the CI spend policy (runs on PRs into main, pushes to main and manual dispatch only; never for
         docs-only changes; superseded runs cancelled), runs the *full* ./bricks.sh (which includes the gate
         self-test), has read-only permissions and a timeout.
  godot: the headless-Godot job pins the same Godot version as the Godot.NET.Sdk in godot/DoraBio.Godot.csproj,
         verifies the download against a pinned SHA-512, and makes gate 8 mandatory (REQUIRE_GODOT=1), so CI
         can never report a skipped Godot test as success.
Exit code 0 on success, 1 with one line per problem otherwise.
"""
import re
import sys

import yaml


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    # YAML 1.1 (PyYAML) reads the bare key `on` as the boolean True.
    if True in doc and "on" not in doc:
        doc["on"] = doc.pop(True)
    return doc


def run_lines(job):
    """All `run:` scripts of a job, joined. Steps without `run` (uses: …) contribute nothing."""
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


def uses(job):
    return [step.get("uses", "") for step in job.get("steps", [])]


def check_core(doc, problems):
    # Spend policy: CI minutes go to merge gates and explicit requests, not to every working-branch push.
    triggers = doc.get("on") or {}
    if set(triggers) != {"push", "pull_request", "workflow_dispatch"}:
        problems.append("triggers must be exactly push, pull_request and workflow_dispatch")
    for event in ("push", "pull_request"):
        spec = triggers.get(event) or {}
        paths = spec.get("paths", [])
        if spec.get("branches") != ["main"]:
            problems.append(f"'{event}' must be limited to branches: [main]")
        if not any(p.startswith("dora-bio/") for p in paths):
            problems.append(f"'{event}' must trigger on paths under dora-bio/")
        if "!dora-bio/**/*.md" not in paths:
            problems.append(f"'{event}' must skip docs-only changes ('!dora-bio/**/*.md')")
    if not (doc.get("concurrency") or {}).get("cancel-in-progress"):
        problems.append("concurrency.cancel-in-progress must be true (superseded runs waste minutes)")
    if doc.get("permissions") != {"contents": "read"}:
        problems.append("top-level permissions must be exactly {contents: read}")

    job = doc.get("jobs", {}).get("gates")
    if not job:
        problems.append("missing job 'gates'")
        return
    if "timeout-minutes" not in job:
        problems.append("job 'gates' needs timeout-minutes")
    if not any(u.startswith("actions/setup-dotnet@") for u in uses(job)):
        problems.append("job 'gates' must set up .NET with actions/setup-dotnet")
    script = run_lines(job)
    # The full run (no brick limit) includes gate 9, and therefore the self-test.
    if not re.search(r"(^|\s)\./bricks\.sh\s*$", script, re.MULTILINE):
        problems.append("job 'gates' must run the full './bricks.sh' (no brick limit)")
    if job.get("defaults", {}).get("run", {}).get("working-directory") != "dora-bio":
        problems.append("job 'gates' must run in working-directory dora-bio")


def check_godot(doc, csproj_path, problems):
    job = doc.get("jobs", {}).get("godot-smoke")
    if not job:
        problems.append("missing job 'godot-smoke'")
        return
    if job.get("needs") != "gates":
        problems.append("job 'godot-smoke' must need 'gates' (no point testing the UI on a red core)")
    env = job.get("env", {})

    with open(csproj_path, encoding="utf-8") as f:
        m = re.search(r'Sdk="Godot\.NET\.Sdk/([0-9.]+)"', f.read())
    sdk_version = m.group(1) if m else None
    if str(env.get("GODOT_VERSION")) != sdk_version:
        problems.append(f"GODOT_VERSION ({env.get('GODOT_VERSION')}) must equal the Godot.NET.Sdk version ({sdk_version})")
    if not re.fullmatch(r"[0-9a-f]{128}", str(env.get("GODOT_SHA512", ""))):
        problems.append("GODOT_SHA512 must be a pinned 128-hex-digit SHA-512")
    if str(env.get("REQUIRE_GODOT")) != "1":
        problems.append("REQUIRE_GODOT must be 1 so a missing Godot fails instead of SKIPPING")

    script = run_lines(job)
    if "sha512sum -c" not in script:
        problems.append("the Godot download must be verified with 'sha512sum -c'")
    if not re.search(r"\./bricks\.sh --only 8", script):
        problems.append("job 'godot-smoke' must run './bricks.sh --only 8'")


def main(argv):
    if len(argv) != 3 or argv[2] not in ("core", "godot"):
        print(__doc__)
        return 64
    problems = []
    try:
        doc = load(argv[1])
    except (OSError, yaml.YAMLError) as e:
        print(f"cannot read workflow: {e}")
        return 1
    if argv[2] == "core":
        check_core(doc, problems)
    else:
        check_godot(doc, "godot/DoraBio.Godot.csproj", problems)
    for p in problems:
        print(f"workflow: {p}")
    if not problems:
        print(f"workflow '{argv[2]}' checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
