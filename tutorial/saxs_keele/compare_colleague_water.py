#!/usr/bin/env python3
"""Compare colleague water profiles with Geant4 using XRD-preprocessing integration."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.ndimage import gaussian_filter1d


HC_EV_M = 1.2398419843320026e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--colleague",
        type=Path,
        required=True,
    )
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--form-factor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--xrd-preprocessing-src",
        type=Path,
        help="Optional source checkout; otherwise use installed xrd-preprocessing",
    )
    parser.add_argument("--pixels", type=int, default=1000)
    return parser.parse_args()


def histogram_hits(
    path: Path, pixels: int, side_mm: float
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    edges = np.linspace(-side_mm / 2.0, side_mm / 2.0, pixels + 1)
    all_image = np.zeros((pixels, pixels), dtype=np.float64)
    rayleigh_image = np.zeros_like(all_image)
    counts = {"all": 0, "rayleigh": 0, "compton": 0, "direct": 0}
    branches = ["posx", "posy", "momz", "type", "trackID", "NRi", "NCi"]
    for arrays in uproot.iterate(
        f"{path}:part", branches, library="np", step_size="160 MB"
    ):
        active = (
            (arrays["type"] == 0)
            & (arrays["trackID"] == 1)
            & (arrays["momz"] > 0.0)
            & (arrays["posx"] >= -side_mm / 2.0)
            & (arrays["posx"] < side_mm / 2.0)
            & (arrays["posy"] >= -side_mm / 2.0)
            & (arrays["posy"] < side_mm / 2.0)
        )
        rayleigh = active & (arrays["NRi"] > 0) & (arrays["NCi"] == 0)
        compton = active & (arrays["NCi"] > 0)
        direct = active & (arrays["NRi"] == 0) & (arrays["NCi"] == 0)
        all_image += np.histogram2d(
            arrays["posy"][active], arrays["posx"][active], bins=(edges, edges)
        )[0]
        rayleigh_image += np.histogram2d(
            arrays["posy"][rayleigh], arrays["posx"][rayleigh], bins=(edges, edges)
        )[0]
        counts["all"] += int(np.count_nonzero(active))
        counts["rayleigh"] += int(np.count_nonzero(rayleigh))
        counts["compton"] += int(np.count_nonzero(compton))
        counts["direct"] += int(np.count_nonzero(direct))
    return all_image, rayleigh_image, counts


def integrate_with_xrd_preprocessing(
    image: np.ndarray,
    *,
    thickness_mm: float,
    base_distance_m: float,
    wavelength_m: float,
    pixel_um: float,
    q_range: tuple[float, float],
    npt: int,
):
    from xrd_preprocessing import AzimuthalIntegration

    pixels = int(image.shape[0])
    frame = pd.DataFrame(
        [
            {
                "measurement_data": image,
                "calculated_distance": base_distance_m,
                "pixel_size": pixel_um,
                "center": (pixels / 2.0, pixels / 2.0),
                "wavelength": wavelength_m * 1.0e9,
                "sample_thickness_mm": thickness_mm,
                "interpolation_q_range": q_range,
            }
        ]
    )
    transformer = AzimuthalIntegration(
        column="measurement_data",
        output_column="radial_profile_data",
        q_range_column="q_range",
        npt=npt,
        mode="1D",
        calibration_mode="dataframe",
        error_model="poisson",
        thickness_adjustment=True,
        require_thickness_adjustment=True,
        thickness_reference_mm=0.0,
        sample_thickness_column="sample_thickness_mm",
    )
    return transformer.fit_transform(frame).iloc[0]


def thickness_remap_q(
    q: np.ndarray, wavelength_m: float, base_distance_m: float, adjusted_distance_m: float
) -> np.ndarray:
    wavelength_nm = wavelength_m * 1.0e9
    theta_base = 2.0 * np.arcsin(
        np.clip(q * wavelength_nm / (4.0 * math.pi), 0.0, 1.0)
    )
    radius_m = base_distance_m * np.tan(theta_base)
    theta_adjusted = np.arctan2(radius_m, adjusted_distance_m)
    return 4.0 * math.pi * np.sin(theta_adjusted / 2.0) / wavelength_nm


def normalize_area(q: np.ndarray, y: np.ndarray, q_low: float = 5.0, q_high: float = 30.0):
    mask = (q >= q_low) & (q <= q_high) & np.isfinite(y)
    area = float(np.trapezoid(np.clip(y[mask], 0.0, None), q[mask]))
    return y / area if area > 0.0 else np.zeros_like(y)


def comparison_metrics(q: np.ndarray, reference: np.ndarray, model: np.ndarray) -> dict[str, float]:
    mask = (q >= 5.0) & (q <= 30.0) & np.isfinite(reference) & np.isfinite(model)
    ref = normalize_area(q, reference)[mask]
    pred = normalize_area(q, model)[mask]
    correlation = float(np.corrcoef(ref, pred)[0, 1])
    relative_rmse = float(
        np.sqrt(np.mean((ref - pred) ** 2)) / np.sqrt(np.mean(ref**2))
    )
    peak_mask = (q >= 10.0) & (q <= 30.0)
    peak_q = float(q[peak_mask][np.argmax(model[peak_mask])])
    return {
        "correlation": correlation,
        "relative_rmse": relative_rmse,
        "model_peak_q_nm^-1": peak_q,
    }


def main() -> None:
    args = parse_args()
    if args.xrd_preprocessing_src is not None:
        sys.path.insert(0, str(args.xrd_preprocessing_src.resolve()))
    colleague = joblib.load(args.colleague)
    if len(colleague) != 3:
        raise ValueError("Expected exactly three colleague profiles.")

    q0 = np.asarray(colleague.iloc[0]["q"], dtype=float)
    q_step = float(np.median(np.diff(q0)))
    q_range = (float(q0[0] - q_step / 2.0), float(q0[-1] + q_step / 2.0))
    wavelength_m = float(colleague.iloc[0]["wavelength"])
    base_distance_m = float(colleague.iloc[0]["dist"])
    theta_corner = 2.0 * math.asin(
        q_range[1] * wavelength_m * 1.0e9 / (4.0 * math.pi)
    )
    corner_radius_mm = base_distance_m * 1.0e3 * math.tan(theta_corner)
    side_mm = math.sqrt(2.0) * corner_radius_mm
    pixel_um = side_mm / args.pixels * 1.0e3

    ff = np.loadtxt(args.form_factor)
    ff_q = ff[:, 0] / 3.8615926744e-4
    ff_s = ff[:, 1] ** 2

    profiles: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)
    colors = {"colleague": "#111827", "g4_all": "#2563eb", "g4_rayleigh": "#dc2626", "keele": "#16a34a"}

    for axis, (_, row) in zip(axes, colleague.iterrows(), strict=True):
        thickness_mm = float(row["thickness"])
        root_path = args.root_dir / f"water_colleague_{int(thickness_mm)}mm_10m.root"
        all_image, rayleigh_image, hit_counts = histogram_hits(
            root_path, args.pixels, side_mm
        )
        integrated_all = integrate_with_xrd_preprocessing(
            all_image,
            thickness_mm=thickness_mm,
            base_distance_m=base_distance_m,
            wavelength_m=wavelength_m,
            pixel_um=pixel_um,
            q_range=q_range,
            npt=q0.size,
        )
        integrated_rayleigh = integrate_with_xrd_preprocessing(
            rayleigh_image,
            thickness_mm=thickness_mm,
            base_distance_m=base_distance_m,
            wavelength_m=wavelength_m,
            pixel_um=pixel_um,
            q_range=q_range,
            npt=q0.size,
        )

        q = np.asarray(integrated_all["q_range"], dtype=float)
        g4_all = np.asarray(integrated_all["radial_profile_data"], dtype=float)
        g4_rayleigh = np.asarray(
            integrated_rayleigh["radial_profile_data"], dtype=float
        )
        colleague_q = np.asarray(row["q"], dtype=float)
        colleague_i = np.asarray(row["I"], dtype=float)
        adjusted_distance_m = float(integrated_all["calculated_distance"])
        colleague_q_adjusted = thickness_remap_q(
            colleague_q, wavelength_m, base_distance_m, adjusted_distance_m
        )
        colleague_original = np.interp(q, colleague_q, colleague_i)
        colleague_adjusted = np.interp(q, colleague_q_adjusted, colleague_i)
        theta = 2.0 * np.arcsin(
            np.clip(q * wavelength_m * 1.0e9 / (4.0 * math.pi), 0.0, 1.0)
        )
        keele = np.interp(q, ff_q, ff_s) * 0.5 * (1.0 + np.cos(theta) ** 2)

        series = {
            "colleague_as_received": colleague_original,
            "colleague_thickness_remap": colleague_adjusted,
            "geant4_all_primary": g4_all,
            "geant4_rayleigh_only": g4_rayleigh,
            "keele_S_times_Thomson": keele,
        }
        for name, values in series.items():
            profiles.append(
                {
                    "thickness_mm": thickness_mm,
                    "series": name,
                    "q_nm^-1": q.copy(),
                    "intensity": values.copy(),
                }
            )

        for reference_name, reference in (
            ("colleague_as_received", colleague_original),
            ("colleague_thickness_remap", colleague_adjusted),
        ):
            for model_name, model in (
                ("geant4_all_primary", g4_all),
                ("geant4_rayleigh_only", g4_rayleigh),
                ("keele_S_times_Thomson", keele),
            ):
                values = comparison_metrics(q, reference, model)
                metrics.append(
                    {
                        "thickness_mm": thickness_mm,
                        "reference": reference_name,
                        "model": model_name,
                        "adjusted_distance_mm": adjusted_distance_m * 1.0e3,
                        **hit_counts,
                        **values,
                    }
                )

        mask = (q >= 5.0) & (q <= 30.0)
        display_sigma_bins = 1.5
        displayed_colleague = gaussian_filter1d(
            normalize_area(q, colleague_original), display_sigma_bins
        )
        displayed_g4_all = gaussian_filter1d(
            normalize_area(q, g4_all), display_sigma_bins
        )
        displayed_g4_rayleigh = gaussian_filter1d(
            normalize_area(q, g4_rayleigh), display_sigma_bins
        )
        all_primary_metrics = comparison_metrics(q, colleague_original, g4_all)
        axis.plot(
            q[mask], displayed_colleague[mask],
            color=colors["colleague"], linewidth=2.0, label="colleague, as received"
        )
        axis.plot(
            q[mask], displayed_g4_all[mask],
            color=colors["g4_all"], linewidth=1.6, label="Geant4, all primary"
        )
        axis.plot(
            q[mask], displayed_g4_rayleigh[mask],
            color=colors["g4_rayleigh"], linewidth=1.4, label="Geant4, Rayleigh only"
        )
        axis.plot(
            q[mask], normalize_area(q, keele)[mask],
            color=colors["keele"], linewidth=1.4, linestyle="--", label=r"Keele $S(q)P(\theta)$"
        )
        axis.set(
            title=f"Water, {thickness_mm:g} mm; corrected L={adjusted_distance_m*1e3:g} mm",
            xlabel=r"$q$ (nm$^{-1}$)",
            ylabel="Area-normalized intensity",
        )
        axis.grid(alpha=0.25)
        axis.text(
            0.03,
            0.96,
            f"G4 all vs colleague: r={all_primary_metrics['correlation']:.3f}, "
            f"rRMSE={100.0*all_primary_metrics['relative_rmse']:.1f}%",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )

    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Colleague water profiles vs Geant4; XRD-preprocessing thickness correction",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)

    profile_path = args.output.with_name(f"{args.output.stem}_profiles.csv")
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["thickness_mm", "series", "q_nm^-1", "intensity"]
        )
        writer.writeheader()
        for entry in profiles:
            for q_value, intensity in zip(
                entry["q_nm^-1"], entry["intensity"], strict=True
            ):
                writer.writerow(
                    {
                        "thickness_mm": entry["thickness_mm"],
                        "series": entry["series"],
                        "q_nm^-1": q_value,
                        "intensity": intensity,
                    }
                )

    metrics_path = args.output.with_name(f"{args.output.stem}_metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    print(f"derived_detector_side_mm={side_mm:.9f}")
    print(f"derived_pixel_um={pixel_um:.9f}")
    for row in metrics:
        print(
            f"t={row['thickness_mm']:g} reference={row['reference']} "
            f"model={row['model']} correlation={row['correlation']:.6f} "
            f"relative_rmse={row['relative_rmse']:.6f} "
            f"peak_q={row['model_peak_q_nm^-1']:.4f}"
        )
    print(f"image={args.output}")
    print(f"profiles={profile_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
