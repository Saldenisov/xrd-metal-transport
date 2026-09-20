#!/usr/bin/env python3
"""Run matched Geant4 and Metal X-ray diffraction examples."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from suite_common import GPU, G4_BINARY, load_cases, run_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    parser.add_argument("--case", action="append", dest="selected",
                        help="Run one named case; repeat to select several")
    parser.add_argument("--photons", type=int, default=100_000_000)
    parser.add_argument("--threads-per-case", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE / "results_100m")
    parser.add_argument("--keep-arrays", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run(task: tuple[dict, int, int, int, Path, bool, bool]) -> dict:
    return run_case(*task)


def main() -> None:
    args = parse_args()
    if min(args.photons, args.threads_per_case, args.workers) < 1:
        raise ValueError("Photons, threads and workers must be positive")
    if args.photons >= 2**32:
        raise ValueError("Metal history index is currently limited to 32 bits")
    if "G4LEDATA" not in os.environ:
        raise RuntimeError("Source geant4.sh so G4LEDATA is available")
    for executable in (G4_BINARY, GPU / "multi_transport"):
        if not executable.exists():
            raise FileNotFoundError(f"Build the repository first: {executable}")

    cases = load_cases(args.cases)
    indexed = list(enumerate(cases))
    if args.selected:
        requested = set(args.selected)
        indexed = [(index, case) for index, case in indexed if case["name"] in requested]
        missing = requested - {case["name"] for _, case in indexed}
        if missing:
            raise ValueError(f"Unknown cases: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "purpose": "Matched Geant4/Metal X-ray diffraction example suite",
        "cases_file": str(args.cases),
        "photons_per_backend_per_case": args.photons,
        "geant4_threads_per_case": args.threads_per_case,
        "parallel_case_workers": args.workers,
        "selected_cases": [case["name"] for _, case in indexed],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    tasks = [
        (case, index, args.photons, args.threads_per_case, args.output,
         args.keep_arrays, args.force)
        for index, case in indexed
    ]
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, task): task[0]["name"] for task in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                print(json.dumps({
                    "case": name,
                    "status": "complete",
                    "g4_seconds": result["geant4_seconds_wall"],
                    "metal_seconds": result["metal_seconds_wall"],
                    "profile_p": result["profile"]["chi2_p_value"],
                }), flush=True)
            except Exception as error:  # Preserve completed cases for resume.
                failures.append((name, repr(error)))
                print(json.dumps({"case": name, "status": "failed",
                                  "error": repr(error)}), flush=True)
    if failures:
        raise RuntimeError(f"Suite failures: {failures}")

    subprocess.run(
        [sys.executable, str(HERE / "compare_suite.py"), "--results", str(args.output)],
        check=True,
    )


if __name__ == "__main__":
    main()
