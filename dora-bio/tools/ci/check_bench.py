#!/usr/bin/env python3
"""Checks a `dora-bio bench --json` report (schema 1), and optionally that it stays within time budgets.

    python3 tools/ci/check_bench.py bench.json [--budget tools/ci/perf_budget.json]

Schema 1: {schema: 1, model_version: str, quick: bool,
           machine: {cores: int > 0, simd_floats: int > 0, simd_accelerated: bool, os: str, dotnet: str},
           timings: [{name, architecture, detail: str, median_seconds >= min_seconds > 0, runs: int >= 1}, ...]}
with exactly one timing for each of the 3 operations × 2 architectures.
Budget file: {budgets: [{name, architecture, max_median_seconds}]}; every budget must match one timing.
Exit 0 when everything holds, 1 otherwise (each problem is printed).
"""
import json
import sys

NAMES = {"denoise", "train-step-n2v", "train-step-supervised"}
ARCHITECTURES = {"dilated", "unet"}


def check_schema(report):
    problems = []

    def need(condition, message):
        if not condition:
            problems.append(message)

    need(isinstance(report, dict), "report must be a JSON object")
    if problems:
        return problems
    need(report.get("schema") == 1, "schema must be 1")
    need(isinstance(report.get("model_version"), str) and report["model_version"].startswith("2."), "model_version must be a 2.x string")
    need(isinstance(report.get("quick"), bool), "quick must be a boolean")
    machine = report.get("machine")
    need(isinstance(machine, dict), "machine must be an object")
    if isinstance(machine, dict):
        for key in ("cores", "simd_floats"):
            need(isinstance(machine.get(key), int) and not isinstance(machine.get(key), bool) and machine[key] > 0, f"machine.{key} must be a positive integer")
        need(isinstance(machine.get("simd_accelerated"), bool), "machine.simd_accelerated must be a boolean")
        for key in ("os", "dotnet"):
            need(isinstance(machine.get(key), str) and machine[key], f"machine.{key} must be a non-empty string")
    timings = report.get("timings")
    need(isinstance(timings, list) and timings, "timings must be a non-empty list")
    seen = set()
    for i, t in enumerate(timings if isinstance(timings, list) else []):
        where = f"timings[{i}]"
        if not isinstance(t, dict):
            problems.append(f"{where} must be an object")
            continue
        need(t.get("name") in NAMES, f"{where}.name must be one of {sorted(NAMES)}")
        need(t.get("architecture") in ARCHITECTURES, f"{where}.architecture must be dilated or unet")
        need(isinstance(t.get("detail"), str) and t["detail"], f"{where}.detail must be a non-empty string")
        median, low, runs = t.get("median_seconds"), t.get("min_seconds"), t.get("runs")
        numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (median, low))
        need(numbers and 0 < low <= median, f"{where}: need 0 < min_seconds <= median_seconds")
        need(isinstance(runs, int) and not isinstance(runs, bool) and runs >= 1, f"{where}.runs must be an integer >= 1")
        key = (t.get("name"), t.get("architecture"))
        need(key not in seen, f"{where}: duplicate timing {key}")
        seen.add(key)
    missing = {(n, a) for n in NAMES for a in ARCHITECTURES} - seen
    need(not missing, f"missing timings: {sorted(missing)}")
    return problems


def check_budget(report, budget):
    problems = []
    timings = {(t["name"], t["architecture"]): t for t in report["timings"]}
    if report.get("quick"):
        problems.append("budgets apply to a full bench run, not --quick")
    for b in budget.get("budgets", []):
        key = (b["name"], b["architecture"])
        if key not in timings:
            problems.append(f"budget {key} has no timing")
            continue
        median = timings[key]["median_seconds"]
        verdict = "within" if median <= b["max_median_seconds"] else "OVER"
        print(f"{b['name']:<22} {b['architecture']:<8} median {median:.4f} s, budget {b['max_median_seconds']:.4f} s: {verdict}")
        if median > b["max_median_seconds"]:
            problems.append(f"{key}: median {median:.4f} s is over the {b['max_median_seconds']} s budget")
    return problems


def main(argv):
    if len(argv) not in (2, 4) or (len(argv) == 4 and argv[2] != "--budget"):
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        report = json.load(f)
    problems = check_schema(report)
    if not problems and len(argv) == 4:
        with open(argv[3], encoding="utf-8") as f:
            problems = check_budget(report, json.load(f))
    for p in problems:
        print(f"✗ {p}")
    if not problems:
        print(f"✓ bench report {'within budget' if len(argv) == 4 else 'matches schema 1'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
