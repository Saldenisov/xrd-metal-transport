#!/usr/bin/env python3
"""Summarize Geant4 SAXS ROOT files and plot Rayleigh events versus q."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


HC_KEV_NM = 1.2398419843320026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--q-max", type=float, default=30.0)
    parser.add_argument("--bins", type=int, default=120)
    return parser.parse_args()


def q_from_theta(energy_kev: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    wavelength_nm = HC_KEV_NM / energy_kev
    return 4.0 * math.pi * np.sin(np.deg2rad(theta_deg) / 2.0) / wavelength_nm


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, args.q_max, args.bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    summaries = []

    figure, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for path in args.root_files:
        with uproot.open(path) as root_file:
            scattering = root_file["scatt"].arrays(library="np")
            particles = root_file["part"].arrays(library="np")

        process = scattering["processID"]
        rayleigh = process == 1
        q_nm = q_from_theta(scattering["e"][rayleigh], scattering["theta"][rayleigh])
        weights = scattering["weight"][rayleigh]
        counts, _ = np.histogram(q_nm, bins=bins, weights=weights)
        normalization = np.trapezoid(counts, centers)
        if normalization > 0:
            counts = counts / normalization
        label = path.stem
        axis.step(centers, counts, where="mid", linewidth=1.4, label=label)

        summaries.append(
            {
                "file": str(path),
                "detector_hits": len(particles["e"]),
                "rayleigh_interactions": int(np.count_nonzero(rayleigh)),
                "compton_interactions": int(np.count_nonzero(process == 2)),
                "photoelectric_interactions": int(np.count_nonzero(process == 3)),
                "rayleigh_q_min_nm-1": float(np.min(q_nm)) if len(q_nm) else math.nan,
                "rayleigh_q_max_nm-1": float(np.max(q_nm)) if len(q_nm) else math.nan,
            }
        )

    axis.set(
        xlabel=r"$q=4\pi\sin(\theta/2)/\lambda$ (nm$^{-1}$)",
        ylabel="Normalized Rayleigh-event density",
        xlim=(0.0, args.q_max),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(args.output_dir / "rayleigh_q_distribution.png", dpi=180)
    plt.close(figure)

    summary_path = args.output_dir / "run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(summary_path)
    print(args.output_dir / "rayleigh_q_distribution.png")


if __name__ == "__main__":
    main()
