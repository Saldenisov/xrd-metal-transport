"""Compare one independently generated Geant4 ROOT to Metal single-Rayleigh output.

Run with eosdx13. ROOT is only an intermediate; figures and JSON are kept.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

from benchmark import DEFAULT_FF, MU_AIR, MU_RAYLEIGH, MU_TOTAL


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--photons", type=int, default=3_000_000)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [str(HERE / "single_rayleigh"), "gpu", str(args.photons),
               "50", "110", "50", "70", "0.1", "320.2",
               str(MU_TOTAL), str(MU_RAYLEIGH), str(MU_AIR),
               "22.162917", "800031", str(DEFAULT_FF)]
    metal = json.loads(subprocess.check_output(command, text=True))
    g4_direct = 0
    g4_rayleigh = 0
    g4_other = 0
    histogram = np.zeros(256, dtype=np.int64)
    edges = np.linspace(0, 70, 257)
    for data in uproot.iterate(
        f"{args.root.resolve()}:part",
        ["e", "posx", "posy", "momz", "type", "trackID", "NRi", "NCi", "NDi"],
        library="np", step_size="100 MB",
    ):
        active = ((data["type"] == 0) & (data["trackID"] == 1)
                  & (data["momz"] > 0) & (np.abs(data["posx"]) < 50)
                  & (np.abs(data["posy"]) < 50))
        direct = active & (data["NRi"] == 0) & (data["NCi"] == 0) & (data["NDi"] == 0)
        rayleigh = active & (data["NRi"] == 1) & (data["NCi"] == 0) & (data["NDi"] == 0)
        g4_direct += int(np.count_nonzero(direct))
        g4_rayleigh += int(np.count_nonzero(rayleigh))
        g4_other += int(np.count_nonzero(active & ~direct & ~rayleigh))
        theta = np.arccos(np.clip(data["momz"][rayleigh], -1, 1))
        q = 4 * np.pi * data["e"][rayleigh] / 1.239841984 * np.sin(theta / 2)
        histogram += np.histogram(q, bins=edges)[0]

    predicted = np.asarray(metal["q_bins"], dtype=float)
    centres = (edges[1:] + edges[:-1]) / 2
    # Independent histories: uncertainty is sqrt(N_G4 + N_Metal).
    sigma = np.sqrt(np.maximum(1, histogram + predicted))
    difference = histogram - predicted
    visible = (centres <= 60) & (histogram + predicted >= 10)
    reduced_chi2 = float(np.mean((difference[visible] / sigma[visible]) ** 2))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    axes[0].step(centres, histogram, where="mid", color="#111827", label="Geant4")
    axes[0].step(centres, predicted, where="mid", color="#ef4444", label="Metal, 1 Rayleigh")
    axes[0].set(ylabel="Photons / q bin", title=f"Independent {args.photons:,}-photon runs; 50 mm slab")
    axes[0].legend()
    axes[1].plot(centres, difference / sigma, color="#2563eb", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].axhline(2, color="gray", linestyle="--", linewidth=0.7)
    axes[1].axhline(-2, color="gray", linestyle="--", linewidth=0.7)
    axes[1].set(xlim=(0, 60), xlabel=r"$q$ (nm$^{-1}$)", ylabel="Residual / σ")
    fig.savefig(args.output_dir / "geant4_vs_metal_single_rayleigh.png", dpi=180)
    plt.close(fig)

    summary = {
        "incident_photons_per_backend": args.photons,
        "g4_direct": g4_direct,
        "metal_direct": metal["classes"][0],
        "g4_single_rayleigh": g4_rayleigh,
        "metal_single_rayleigh": metal["classes"][1],
        "g4_other_detected": g4_other,
        "q_reduced_chi2_independent_samples": reduced_chi2,
        "q_compared_bins": int(np.count_nonzero(visible)),
        "important_limit": "Metal does not transport Compton or multiple collisions; only direct and exactly-one-Rayleigh channels are compared.",
    }
    (args.output_dir / "geant4_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
