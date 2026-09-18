"""Reproducible 100-geometry CPU/Metal parity and timing experiment.

Use eosdx13. The transport kernel models zero or exactly one Rayleigh event;
other first interactions and subsequent interactions are counted, not tracked.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_FF = HERE.parent / "data" / "keele_breast_g50.dat"
MU_TOTAL = 0.0250599 + 0.018054 + 0.00651873
MU_RAYLEIGH = 0.00651873
MU_AIR = 4.15581e-5 + 2.07728e-5 + 9.58023e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--photons", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--ff", type=Path, default=DEFAULT_FF)
    parser.add_argument("--binary", type=Path, default=HERE / "single_rayleigh")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def run_case(
    binary: Path, ff: Path, mode: str, photons: int, values: dict[str, float], seed: int
) -> dict:
    command = [
        str(binary.resolve()),
        mode,
        str(photons),
        str(values["thickness_mm"]),
        str(values["gap_mm"]),
        "50",  # half-width: 1000 x 1000 at 100 um
        "70",  # q_nm^-1 histogram range
        str(values["beam_radius_mm"]),
        str(values["focus_from_entry_mm"]),
        str(values["mu_total_mm_inv"]),
        str(values["mu_rayleigh_mm_inv"]),
        str(MU_AIR),
        "22.162917",  # Ag K-alpha1
        str(seed),
        str(ff.resolve()),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def main() -> None:
    args = parse_args()
    if args.cases < 1 or args.photons < 1:
        raise ValueError("cases and photons must be positive")
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(args.cases):
        thickness = float(rng.uniform(10, 60))
        gap = float(rng.uniform(60, 180))
        density = float(rng.uniform(0.8, 1.2))
        values = {
            "thickness_mm": thickness,
            "gap_mm": gap,
            "beam_radius_mm": float(rng.uniform(0.05, 0.15)),
            "focus_from_entry_mm": float(rng.uniform(280, 450)),
            "mu_total_mm_inv": MU_TOTAL * density,
            "mu_rayleigh_mm_inv": MU_RAYLEIGH * density,
            "density_scale": density,
        }
        case_seed = args.seed + index * 17
        cpu = run_case(args.binary, args.ff, "cpu", args.photons, values, case_seed)
        gpu = run_case(args.binary, args.ff, "gpu", args.photons, values, case_seed)
        record = {
            "case": index,
            **values,
            "cpu_s": cpu["seconds"],
            "metal_s": gpu["seconds"],
            "speedup": cpu["seconds"] / gpu["seconds"],
            "class_l1": sum(abs(a - b) for a, b in zip(cpu["classes"], gpu["classes"])),
            "q_profile_l1": sum(abs(a - b) for a, b in zip(cpu["q_bins"], gpu["q_bins"])),
            "cpu_direct": cpu["classes"][0],
            "cpu_single_rayleigh": cpu["classes"][1],
            "cpu_first_other": cpu["classes"][2],
            "cpu_detector_miss": cpu["classes"][3],
            "cpu_second_interaction": cpu["classes"][4],
            "cpu_air_interaction": cpu["classes"][5],
        }
        records.append(record)
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{args.cases} cases", flush=True)

    with (args.output_dir / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    speeds = np.asarray([record["speedup"] for record in records])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.scatter(np.arange(args.cases), speeds, s=18, color="#2563eb")
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Geometry index", ylabel="CPU time / Metal time",
           title=f"Single-Rayleigh transport; {args.photons:,} photons per geometry")
    fig.savefig(args.output_dir / "speedup.png", dpi=170)
    plt.close(fig)
    summary = {
        "cases": args.cases,
        "photons_per_case": args.photons,
        "all_class_counts_equal": all(x["class_l1"] == 0 for x in records),
        "all_q_histograms_equal": all(x["q_profile_l1"] == 0 for x in records),
        "cases_with_rounding_difference": sum(
            x["class_l1"] > 0 or x["q_profile_l1"] > 0 for x in records
        ),
        "total_class_l1_over_all_cases": sum(x["class_l1"] for x in records),
        "total_q_histogram_l1_over_all_cases": sum(x["q_profile_l1"] for x in records),
        "max_class_l1_per_case": max(x["class_l1"] for x in records),
        "max_q_histogram_l1_per_case": max(x["q_profile_l1"] for x in records),
        "median_speedup": float(np.median(speeds)),
        "mean_cpu_s": float(np.mean([x["cpu_s"] for x in records])),
        "mean_metal_s": float(np.mean([x["metal_s"] for x in records])),
        "physics_scope": "single-Rayleigh only; other first and subsequent interactions not transported",
        "cross_sections_source": "Geant4 SAXS_DUMP_XS at 22.162917 keV for g50 material and Air",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
