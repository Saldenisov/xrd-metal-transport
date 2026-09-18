"""Time Geant4 and Metal on the same 50 mm g50 detector-image workload."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from benchmark_g4 import BUILD, DEFAULT_FF, G4_BINARY, HERE, make_macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photons", type=int, default=20_000_000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="keele_metal_fair_benchmark_") as temporary:
        stem = Path(temporary) / "g50"
        case = dict(thickness_mm="50", gap_mm="110", beam_radius_mm="0.1",
                    focus_from_entry_mm="320", density_scale="1", case="0")
        macro = stem.with_suffix(".mac")
        macro.write_text(make_macro(case, args.photons, stem))
        environment = os.environ.copy()
        environment["SAXS_IMAGE_ONLY"] = "1"
        environment["SAXS_IMAGE_CSV"] = "1"
        environment["SAXS_IMAGE_PIXELS"] = "1000"
        environment["SAXS_IMAGE_PITCH_UM"] = "100"
        environment.pop("SAXS_DUMP_XS", None)
        started = time.perf_counter()
        g4 = subprocess.run([str(G4_BINARY), str(macro), str(args.threads)],
                            cwd=BUILD, env=environment, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
        g4_wall = time.perf_counter() - started
        if g4.returncode:
            raise RuntimeError(g4.stdout[-4000:])
        if stem.with_suffix(".root").exists():
            raise AssertionError("Image-only Geant4 unexpectedly created ROOT")
        raw = Path(temporary) / "image.raw"
        command = [str(HERE / "multi_transport"), str(args.photons), "50", "110",
                   "50", "0.1", "0.1", "320", "1", "1", "1", "1", "42091",
                   str(HERE / "results" / "multi_physics_g50.json"), str(DEFAULT_FF), str(raw)]
        started = time.perf_counter()
        metal = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
        metal_wall = time.perf_counter() - started
        if metal.returncode:
            raise RuntimeError(metal.stderr)
        result = {"photons": args.photons, "g4_threads": args.threads,
                  "g4_image_only_csv_wall_seconds": g4_wall,
                  "metal_image_raw_wall_seconds": metal_wall,
                  "speedup_wall": g4_wall / metal_wall,
                  "metal_transport_and_histogram_seconds": json.loads(metal.stdout)["seconds_total"],
                  "root_files_retained": 0,
                  "caveat": "Geant4 CSV and Metal raw have different serialization costs; "
                            "both timings include process startup and complete detector-image transport"}
    output = HERE / "results" / "multi_transport" / "fair_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
