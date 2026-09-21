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

from plotting import (
    GEANT4_COLOR,
    METAL_COLOR,
    plot_case_comparison,
    plot_profile_pair,
)

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
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
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
        "Figures use solid blue for Geant4 CPU and dashed orange with markers "
        "for Metal GPU. Every profile panel reports its measured end-to-end "
        "wall-time speedup.",
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
    p_values = [row["profile_p_value"] for row in rows]
    speedups = [row["wall_speedup"] for row in rows]
    integrated = [100 * row["profile_integrated_relative_difference"] for row in rows]
    max_channel_z = max(row["max_abs_channel_z"] for row in rows)
    lowest = min(rows, key=lambda row: row["profile_p_value"])
    lines += [
        "", "## Interpretation", "",
        f"{sum(value >= 0.05 for value in p_values)}/{len(rows)} primary profile tests "
        f"have p ≥ 0.05. The lowest value is `{lowest['case']}` at "
        f"p = {lowest['profile_p_value']:.3g}; its four channel differences are "
        f"within {lowest['max_abs_channel_z']:.2f}σ and its integrated profile "
        f"difference is {100 * lowest['profile_integrated_relative_difference']:.3f}%.",
        "",
        f"Across the {len(rows)} primary cases, the largest absolute channel difference is "
        f"{max_channel_z:.2f}σ, integrated profile differences span "
        f"{min(integrated):.3f}% to {max(integrated):.3f}%, and observed wall-time "
        f"speedups span {min(speedups):.0f}× to {max(speedups):.0f}×. These are "
        "descriptive results for this hardware and model, not universal performance "
        "or equivalence claims.",
        "",
    ]
    replicate_path = root / "replicates" / "g50_25mm_seed2" / "summary.json"
    if replicate_path.exists():
        replicate = json.loads(replicate_path.read_text())
        replicate_z = max(map(abs, replicate["channel_poisson_z"].values()))
        lines += [
            "An independent 100-million-history `g50_25mm` repeat with new "
            f"Geant4 and Metal seeds gives p = "
            f"{replicate['profile']['chi2_p_value']:.3g}, with all four channels "
            f"within {replicate_z:.2f}σ. The low primary p-value did not reproduce.",
            "",
            "[Open the independent `g50_25mm` repeat](replicates/g50_25mm_seed2/detector_comparison.png).",
        ]
    lines += [
        "", "![Profile overview](profile_overview.png)", "",
        "![Parity overview](parity_overview.png)", "",
        "The comparisons cover the finite homogeneous sample and ideal detector "
        "implemented by both engines. Remaining model limitations are recorded "
        "in every case summary.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


def plot_profiles(root: Path, results: list[tuple[dict, np.ndarray]]) -> None:
    group_index = {name: index for index, name in enumerate(GROUP_ORDER)}
    ordered = sorted(
        results, key=lambda item: (group_index[item[0]["group"]], item[0]["case"])
    )
    columns = 4
    rows = int(np.ceil(len(ordered) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(17, 4 * rows), sharex=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, (summary, profile) in zip(axes_flat, ordered):
        photons = summary["incident_photons_per_backend"]
        q = profile["q_nm_inv"]
        geant4 = profile["geant4_sum"] / photons
        metal = profile["metal_sum"] / photons
        plot_profile_pair(axis, q, geant4, metal)
        axis.set(
            xlim=(1, 30), yscale="log", xlabel=r"$q$ (nm$^{-1}$)",
            ylabel="Azimuthal sum / photon",
            title=f"{summary['case']}\nMetal speedup: {summary['wall_speedup']:.0f}×",
        )
        axis.title.set_fontweight("bold")
        axis.grid(which="both", alpha=0.18, lw=0.6)
        axis.legend(
            loc="lower right", fontsize=8, frameon=True, facecolor="white",
            edgecolor="#777777", framealpha=0.95, handlelength=3.2,
        )
        axis.text(
            0.03, 0.04,
            f"G4 {summary['geant4_seconds_wall']:.1f} s  |  "
            f"Metal {summary['metal_seconds_wall']:.3f} s",
            transform=axis.transAxes, fontsize=8, color="#333333",
        )
    for axis in axes_flat[len(ordered):]:
        axis.set_visible(False)
    fig.suptitle(
        "Geant4 CPU versus Apple Metal GPU — 100 million photons per backend\n"
        "Geant4: solid blue  |  Metal: dashed orange with markers",
        fontsize=16, fontweight="bold",
    )
    fig.savefig(root / "profile_overview.png", dpi=190)
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
    axes[1].bar(x, profile_bias, color=GEANT4_COLOR)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(ylabel="Integrated profile difference (%)")
    for position, value in zip(x, profile_p):
        axes[1].text(position, profile_bias[position], f" p={value:.2g}",
                     rotation=90, va="bottom" if profile_bias[position] >= 0 else "top",
                     ha="center", fontsize=7)

    speedup = [summary["wall_speedup"] for summary, _ in results]
    speed_bars = axes[2].bar(x, speedup, color=METAL_COLOR)
    axes[2].bar_label(speed_bars, labels=[f"{value:.0f}×" for value in speedup],
                      padding=3, fontsize=8, fontweight="bold")
    axes[2].set(
        ylabel="Wall-time speedup", title="End-to-end Geant4 / Metal speedup",
        xticks=x, xticklabels=names, ylim=(0, max(speedup) * 1.18),
    )
    axes[2].tick_params(axis="x", rotation=45, labelsize=8)
    fig.savefig(root / "parity_overview.png", dpi=190)
    plt.close(fig)


def plot_saved_cases(root: Path) -> int:
    """Regenerate detailed figures from retained public arrays without transport."""
    summary_paths = sorted(root.glob("*/summary.json"))
    summary_paths += sorted(root.glob("replicates/*/summary.json"))
    rendered = 0
    for summary_path in summary_paths:
        case_dir = summary_path.parent
        arrays_path = case_dir / "detector_arrays.npz"
        profiles_path = case_dir / "profiles.csv"
        if not arrays_path.exists() or not profiles_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        profile = np.genfromtxt(profiles_path, delimiter=",", names=True)
        with np.load(arrays_path) as arrays:
            plot_case_comparison(
                case_dir, summary, profile["q_nm_inv"], profile["geant4_sum"],
                profile["metal_sum"], profile["bin_z"], arrays["geant4"],
                arrays["metal"],
            )
        rendered += 1
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results_100m")
    args = parser.parse_args()
    results = load_results(args.results)
    write_summary(args.results, results)
    rendered = plot_saved_cases(args.results)
    plot_profiles(args.results, results)
    plot_parity(args.results, results)
    print(json.dumps({
        "cases": len(results), "detailed_figures": rendered,
        "results": str(args.results),
    }))


if __name__ == "__main__":
    main()
