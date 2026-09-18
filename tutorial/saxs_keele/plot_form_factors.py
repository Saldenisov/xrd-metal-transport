#!/usr/bin/env python3
"""Plot Keele component intensities against the Geant4 reference tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from prepare_keele_ff import (
    COMPONENTS,
    ELECTRON_REDUCED_COMPTON_NM,
    read_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("../../../xrd_components.txt"))
    parser.add_argument(
        "--geant4-miff-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "install"
            / "geant4-11.4.2"
            / "share"
            / "Geant4"
            / "data"
            / "G4EMLOW8.8"
            / "penelope"
            / "rayleigh"
            / "MIFF"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("results/form_factors.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measured = read_components(args.input.resolve())
    colors = {"fat": "#d97706", "water": "#2563eb", "collagen": "#7c3aed"}
    figure, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)

    for name, metadata in COMPONENTS.items():
        q_measured, intensity = zip(*measured[name])
        axis.plot(q_measured, intensity, color=colors[name], linewidth=1.7, label=f"Keele {name}")

        q_reference = []
        intensity_reference = []
        with (args.geant4_miff_dir / str(metadata["reference"])).open() as stream:
            for line in stream:
                q_geant4, form_factor = map(float, line.split())
                q_nm = q_geant4 / ELECTRON_REDUCED_COMPTON_NM
                if q_nm > 35.0:
                    break
                q_reference.append(q_nm)
                intensity_reference.append(form_factor * form_factor)
        axis.plot(
            q_reference,
            intensity_reference,
            color=colors[name],
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label=f"Geant4 {name}",
        )

    axis.set(
        xlabel=r"$q$ (nm$^{-1}$)",
        ylabel=r"$F^2(q)/W$",
        xlim=(0.0, 30.0),
        ylim=(0.0, None),
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
