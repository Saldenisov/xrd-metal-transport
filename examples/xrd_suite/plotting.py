"""High-contrast figures for Geant4 and Metal XRD comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GEANT4_COLOR = "#0057B8"
METAL_COLOR = "#E65100"
RESIDUAL_COLOR = "#6A3D9A"
Q_RANGE = (1.0, 30.0)


def plot_profile_pair(axis, q: np.ndarray, geant4: np.ndarray,
                      metal: np.ndarray, *, labels: bool = True) -> None:
    """Draw one pair so both nearly coincident curves remain visible."""
    axis.plot(
        q, geant4, color=GEANT4_COLOR, lw=3.0, solid_capstyle="round",
        label="Geant4 CPU" if labels else None, zorder=2,
    )
    axis.plot(
        q, metal, color=METAL_COLOR, lw=2.4, ls=(0, (5, 2)),
        marker="o", markevery=16, ms=3.2, mfc="white", mew=1.0,
        label="Metal GPU" if labels else None, zorder=3,
    )


def plot_case_comparison(output: Path, summary: dict, q: np.ndarray,
                         geant4_profile: np.ndarray,
                         metal_profile: np.ndarray, bin_z: np.ndarray,
                         geant4_image: np.ndarray,
                         metal_image: np.ndarray) -> None:
    """Render detector maps, profiles, residuals and timing for one case."""
    photons = summary["incident_photons_per_backend"]
    geant4_profile = geant4_profile / photons
    metal_profile = metal_profile / photons
    speedup = summary["wall_speedup"]
    geant4_seconds = summary["geant4_seconds_wall"]
    metal_seconds = summary["metal_seconds_wall"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    extent = (-50, 50, -50, 50)
    maximum = max(
        np.percentile(np.log1p(geant4_image), 99.99),
        np.percentile(np.log1p(metal_image), 99.99),
    )
    map_titles = (
        f"Geant4 CPU — {geant4_seconds:.1f} s",
        f"Metal GPU — {metal_seconds:.3f} s",
    )
    for axis, image, title in zip(
        axes[0], (geant4_image, metal_image), map_titles
    ):
        axis.imshow(
            np.log1p(image), origin="lower", extent=extent, cmap="magma",
            vmin=0, vmax=maximum,
        )
        axis.set(
            xlim=(-25, 25), ylim=(-25, 25), xlabel="x (mm)", ylabel="y (mm)",
            title=f"{title}\nlog(1 + counts)",
        )

    plot_profile_pair(axes[1, 0], q, geant4_profile, metal_profile)
    axes[1, 0].set(
        xlim=Q_RANGE, yscale="log", xlabel=r"$q$ (nm$^{-1}$)",
        ylabel="Azimuthal sum / incident photon",
    )
    axes[1, 0].grid(which="both", alpha=0.18, lw=0.6)
    axes[1, 0].legend(
        frameon=True, facecolor="white", edgecolor="#777777",
        framealpha=0.95, handlelength=3.5,
    )
    axes[1, 0].text(
        0.04, 0.06, f"End-to-end speedup: {speedup:.0f}×",
        transform=axes[1, 0].transAxes, fontsize=10, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF3E0",
              "edgecolor": METAL_COLOR, "alpha": 0.95},
    )

    axes[1, 1].plot(q, bin_z, lw=1.8, color=RESIDUAL_COLOR)
    axes[1, 1].axhline(0, color="black", lw=1.0)
    axes[1, 1].axhline(3, color="#555555", lw=1.0, ls="--")
    axes[1, 1].axhline(-3, color="#555555", lw=1.0, ls="--")
    axes[1, 1].set(
        xlim=Q_RANGE, xlabel=r"$q$ (nm$^{-1}$)",
        ylabel=r"(Metal $-$ Geant4) / $\sigma$",
        title="Bin-wise statistical residual",
    )
    axes[1, 1].grid(alpha=0.18, lw=0.6)

    fractions = summary["mass_fractions_water_fat_collagen"]
    thickness = summary["geometry"]["sample_thickness_mm"]
    figure_title = (
        f"{summary['case']}: Geant4 CPU versus Metal GPU — {speedup:.0f}× faster\n"
        f"{thickness:g} mm; water/fat/collagen = "
        f"{fractions['water']:.3g}/{fractions['fat']:.3g}/"
        f"{fractions['collagen']:.3g}; {photons:,} photons per backend"
    )
    fig.suptitle(figure_title, fontsize=14, fontweight="bold")
    fig.savefig(output / "detector_comparison.png", dpi=190)
    plt.close(fig)
