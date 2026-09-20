#!/usr/bin/env python3
"""Aggregate completed X-ray diffraction suite results and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CHANNELS = ("direct", "single_rayleigh", "multiple_rayleigh", "compton")
GROUP_ORDER = (
    "pure components", "binary mixtures", "ternary mixtures", "g50 thickness series"
)


def load_results(root: Path) -> list[tuple[dict, np.ndarray]]:
    results = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        profile = np.genfromtxt(
            summary_path.with_name("profiles.csv"), delimiter=",", names=True
        )
        results.append((summary, profile))
    if not results:
        raise FileNotFoundError(f"No completed case results in {root}")
    return results


def write_summary(root: Path, results: list[tuple[dict, np.ndarray]]) -> None:
    fields = [
        "case", "group", "thickness_mm", "water", "fat", "collagen", "photons",
        "geant4_seconds", "metal_seconds", "wall_speedup", "profile_p_value",
        "profile_reduced_chi2", "profile_integrated_relative_difference",
        "normalized_shape_relative_l2", "max_abs_channel_z",
    ]
    rows = []
    for summary, _ in results:
        fractions = summary["mass_fractions_water_fat_collagen"]
        rows.append({
            "case": summary["case"],
            "group": summary["group"],
            "thickness_mm": summary["geometry"]["sample_thickness_mm"],
            "water": fractions["water"], "fat": fractions["fat"],
            "collagen": fractions["collagen"],
            "photons": summary["incident_photons_per_backend"],
            "geant4_seconds": summary["geant4_seconds_wall"],
            "metal_seconds": summary["metal_seconds_wall"],
            "wall_speedup": summary["wall_speedup"],
            "profile_p_value": summary["profile"]["chi2_p_value"],
            "profile_reduced_chi2": summary["profile"]["reduced_chi2"],
            "profile_integrated_relative_difference":
                summary["profile"]["integrated_relative_difference"],
            "normalized_shape_relative_l2":
                summary["profile"]["normalized_shape_relative_l2_2_to_28_nm_inv"],
            "max_abs_channel_z": max(map(abs, summary["channel_poisson_z"].values())),
        })
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    photon_counts = {row["photons"] for row in rows}
    count_label = (f"{next(iter(photon_counts)):,}" if len(photon_counts) == 1
                   else "case-specific")
    lines = [
        "# Geant4 versus Metal: X-ray diffraction examples",
        "",
        f"Each row compares {count_label} independent histories per backend. "
        "Profile p-values use the full pyFAI pixel-splitting covariance.",
        "",
        "| Case | Water/fat/collagen | Thickness | Geant4 | Metal | Speedup | "
        "Profile χ²/ν | p | Figure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        fractions = f"{row['water']:.3g}/{row['fat']:.3g}/{row['collagen']:.3g}"
        lines.append(
            f"| `{row['case']}` | {fractions} | {row['thickness_mm']:g} mm | "
            f"{row['geant4_seconds']:.1f} s | {row['metal_seconds']:.3f} s | "
            f"{row['wall_speedup']:.0f}× | {row['profile_reduced_chi2']:.3f} | "
            f"{row['profile_p_value']:.3g} | "
            f"[plot]({row['case']}/detector_comparison.png) |"
        )
    lines += [
        "", "![Profile overview](profile_overview.png)", "",
        "![Parity overview](parity_overview.png)", "",
        "The comparisons cover the finite homogeneous sample and ideal detector "
        "implemented by both engines. Remaining model limitations are recorded "
        "in every case summary.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


def plot_profiles(root: Path, results: list[tuple[dict, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, group in zip(axes.ravel(), GROUP_ORDER):
        selected = [(summary, profile) for summary, profile in results
                    if summary["group"] == group]
        for index, (summary, profile) in enumerate(selected):
            q = profile["q_nm_inv"]
            g4 = profile["geant4_sum"] / summary["incident_photons_per_backend"]
            metal = profile["metal_sum"] / summary["incident_photons_per_backend"]
            line = axis.plot(q, g4, lw=1.4, label=summary["case"])[0]
            axis.plot(q, metal, lw=0.9, ls="--", color=line.get_color())
        axis.set(title=group, xlim=(1, 30), yscale="log",
                 xlabel=r"$q$ (nm$^{-1}$)", ylabel="Azimuthal sum / photon")
        if selected:
            axis.legend(fontsize=7, frameon=False)
    fig.suptitle("X-ray diffraction profiles: Geant4 solid, Metal dashed")
    fig.savefig(root / "profile_overview.png", dpi=170)
    plt.close(fig)


def plot_parity(root: Path, results: list[tuple[dict, np.ndarray]]) -> None:
    names = [summary["case"] for summary, _ in results]
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             constrained_layout=True)
    width = 0.18
    for offset, channel in enumerate(CHANNELS):
        values = [100 * summary["channel_relative_difference"][channel]
                  for summary, _ in results]
        axes[0].bar(x + (offset - 1.5) * width, values, width=width, label=channel)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(ylabel="Metal / Geant4 − 1 (%)", title="Detector channel differences")
    axes[0].legend(ncol=4, fontsize=8, frameon=False)

    profile_bias = [100 * summary["profile"]["integrated_relative_difference"]
                    for summary, _ in results]
    profile_p = [summary["profile"]["chi2_p_value"] for summary, _ in results]
    axes[1].bar(x, profile_bias, color="#477fa8")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(ylabel="Integrated profile difference (%)")
    for position, value in zip(x, profile_p):
        axes[1].text(position, profile_bias[position], f" p={value:.2g}",
                     rotation=90, va="bottom" if profile_bias[position] >= 0 else "top",
                     ha="center", fontsize=7)

    speedup = [summary["wall_speedup"] for summary, _ in results]
    axes[2].bar(x, speedup, color="#38866f")
    axes[2].set(ylabel="Wall-time speedup", xticks=x, xticklabels=names)
    axes[2].tick_params(axis="x", rotation=45, labelsize=8)
    fig.savefig(root / "parity_overview.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results_100m")
    args = parser.parse_args()
    results = load_results(args.results)
    write_summary(args.results, results)
    plot_profiles(args.results, results)
    plot_parity(args.results, results)
    print(json.dumps({"cases": len(results), "results": str(args.results)}))


if __name__ == "__main__":
    main()
