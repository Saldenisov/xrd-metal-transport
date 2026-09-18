#!/usr/bin/env python3
"""Reintegrate the 50 mm breast simulation with thickness-corrected geometry."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import uproot


HC_EV_M = 1.2398419843320026e-6
ELECTRON_REDUCED_COMPTON_NM = 3.8615926744e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--form-factor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=1000)
    parser.add_argument("--pitch-um", type=float, default=100.0)
    parser.add_argument("--energy-kev", type=float, default=22.162917)
    parser.add_argument("--sample-thickness-mm", type=float, default=50.0)
    parser.add_argument(
        "--downstream-gap-mm",
        type=float,
        default=120.0,
        help="Distance from the downstream sample face to the detector plane.",
    )
    parser.add_argument("--q-min", type=float, default=1.0)
    parser.add_argument("--q-max", type=float, default=35.0)
    parser.add_argument("--radial-points", type=int, default=100)
    parser.add_argument(
        "--xrd-preprocessing-src", type=Path,
        help="Optional source checkout; otherwise use installed xrd-preprocessing"
    )
    return parser.parse_args()


def normalize_area(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = (q >= 5.0) & (q <= 35.0) & np.isfinite(y)
    clipped = np.clip(np.asarray(y, dtype=float), 0.0, None)
    area = float(np.trapezoid(clipped[mask], q[mask]))
    return clipped / area if area > 0.0 else np.zeros_like(clipped)


def metrics(q: np.ndarray, reference: np.ndarray, model: np.ndarray) -> dict[str, float]:
    mask = (q >= 5.0) & (q <= 35.0) & np.isfinite(reference) & np.isfinite(model)
    ref = normalize_area(q, reference)[mask]
    pred = normalize_area(q, model)[mask]
    peak_mask = (q >= 5.0) & (q <= 35.0)
    return {
        "correlation": float(np.corrcoef(ref, pred)[0, 1]),
        "relative_rmse": float(
            np.sqrt(np.mean((ref - pred) ** 2)) / np.sqrt(np.mean(ref**2))
        ),
        "reference_peak_q_nm^-1": float(
            q[peak_mask][np.argmax(reference[peak_mask])]
        ),
        "model_peak_q_nm^-1": float(q[peak_mask][np.argmax(model[peak_mask])]),
    }


def load_detector_and_truth(
    path: Path,
    *,
    pixels: int,
    pitch_um: float,
    energy_kev: float,
    q_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    side_mm = pixels * pitch_um * 1e-3
    image_edges = np.linspace(-side_mm / 2.0, side_mm / 2.0, pixels + 1)
    single_rayleigh_image = np.zeros((pixels, pixels), dtype=np.float64)
    all_rayleigh_image = np.zeros_like(single_rayleigh_image)
    truth_counts = np.zeros(q_edges.size - 1, dtype=np.int64)
    counts = {
        "all_primary": 0,
        "direct": 0,
        "single_rayleigh": 0,
        "all_rayleigh": 0,
        "compton": 0,
    }
    branches = ["e", "posx", "posy", "momz", "type", "trackID", "NRi", "NCi"]
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
        direct = active & (arrays["NRi"] == 0) & (arrays["NCi"] == 0)
        single_rayleigh = active & (arrays["NRi"] == 1) & (arrays["NCi"] == 0)
        all_rayleigh = active & (arrays["NRi"] > 0) & (arrays["NCi"] == 0)
        compton = active & (arrays["NCi"] > 0)

        single_rayleigh_image += np.histogram2d(
            arrays["posy"][single_rayleigh],
            arrays["posx"][single_rayleigh],
            bins=(image_edges, image_edges),
        )[0]
        all_rayleigh_image += np.histogram2d(
            arrays["posy"][all_rayleigh],
            arrays["posx"][all_rayleigh],
            bins=(image_edges, image_edges),
        )[0]

        theta = np.arccos(np.clip(arrays["momz"][single_rayleigh], -1.0, 1.0))
        wavelength_nm = HC_EV_M * 1e9 / (arrays["e"][single_rayleigh] * 1000.0)
        truth_q = 4.0 * math.pi * np.sin(theta / 2.0) / wavelength_nm
        truth_counts += np.histogram(truth_q, bins=q_edges)[0]

        counts["all_primary"] += int(np.count_nonzero(active))
        counts["direct"] += int(np.count_nonzero(direct))
        counts["single_rayleigh"] += int(np.count_nonzero(single_rayleigh))
        counts["all_rayleigh"] += int(np.count_nonzero(all_rayleigh))
        counts["compton"] += int(np.count_nonzero(compton))

    return single_rayleigh_image, all_rayleigh_image, truth_counts, counts


def integrate_with_xrd_preprocessing(
    image: np.ndarray,
    *,
    pixel_um: float,
    wavelength_m: float,
    front_face_distance_m: float,
    sample_thickness_mm: float,
    q_range: tuple[float, float],
    npt: int,
    correct_solid_angle: bool | None = None,
):
    from xrd_preprocessing import AzimuthalIntegration

    pixels = int(image.shape[0])
    frame = pd.DataFrame(
        [
            {
                "measurement_data": image,
                "calculated_distance": front_face_distance_m,
                "pixel_size": pixel_um,
                "center": (pixels / 2.0, pixels / 2.0),
                "wavelength": wavelength_m * 1e9,
                "sample_thickness_mm": sample_thickness_mm,
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
        correct_solid_angle=correct_solid_angle,
        thickness_adjustment=True,
        require_thickness_adjustment=True,
        thickness_reference_mm=0.0,
        sample_thickness_column="sample_thickness_mm",
    )
    return transformer.fit_transform(frame).iloc[0]


def input_form_factor(
    path: Path, q_nm: np.ndarray, wavelength_m: float
) -> np.ndarray:
    table = np.loadtxt(path)
    table_q_nm = table[:, 0] / ELECTRON_REDUCED_COMPTON_NM
    f2_over_w = table[:, 1] ** 2
    wavelength_nm = wavelength_m * 1e9
    theta = 2.0 * np.arcsin(
        np.clip(q_nm * wavelength_nm / (4.0 * math.pi), 0.0, 1.0)
    )
    thomson = 0.5 * (1.0 + np.cos(theta) ** 2)
    return np.interp(q_nm, table_q_nm, f2_over_w) * thomson


def main() -> None:
    args = parse_args()
    if args.xrd_preprocessing_src is not None:
        sys.path.insert(0, str(args.xrd_preprocessing_src.resolve()))

    wavelength_m = HC_EV_M / (args.energy_kev * 1000.0)
    q_edges = np.linspace(args.q_min, args.q_max, args.radial_points + 1)
    truth_q = 0.5 * (q_edges[:-1] + q_edges[1:])
    (
        single_rayleigh_image,
        all_rayleigh_image,
        truth_counts,
        hit_counts,
    ) = load_detector_and_truth(
        args.root,
        pixels=args.pixels,
        pitch_um=args.pitch_um,
        energy_kev=args.energy_kev,
        q_edges=q_edges,
    )

    # XRD-preprocessing convention: zero-thickness reference at the upstream
    # sample face. The transformer subtracts half the 50 mm sample thickness,
    # yielding the mean scattering-plane distance: 170 - 25 = 145 mm.
    front_face_distance_m = (
        args.downstream_gap_mm + args.sample_thickness_mm
    ) * 1e-3
    corrected = integrate_with_xrd_preprocessing(
        single_rayleigh_image,
        pixel_um=args.pitch_um,
        wavelength_m=wavelength_m,
        front_face_distance_m=front_face_distance_m,
        sample_thickness_mm=args.sample_thickness_mm,
        q_range=(args.q_min, args.q_max),
        npt=args.radial_points,
    )
    corrected_all_rayleigh = integrate_with_xrd_preprocessing(
        all_rayleigh_image,
        pixel_um=args.pitch_um,
        wavelength_m=wavelength_m,
        front_face_distance_m=front_face_distance_m,
        sample_thickness_mm=args.sample_thickness_mm,
        q_range=(args.q_min, args.q_max),
        npt=args.radial_points,
    )

    q = np.asarray(corrected["q_range"], dtype=float)
    detector_profile = np.asarray(corrected["radial_profile_data"], dtype=float)
    all_rayleigh_profile = np.asarray(
        corrected_all_rayleigh["radial_profile_data"], dtype=float
    )
    corrected_distance_m = float(corrected["calculated_distance"])
    truth_profile = np.interp(q, truth_q, truth_counts / np.maximum(truth_q, 1e-12))
    ff_profile = input_form_factor(args.form_factor, q, wavelength_m)

    comparisons = {
        "xrd_preprocessing_vs_geant4_truth": metrics(q, truth_profile, detector_profile),
        "xrd_preprocessing_vs_F2_Thomson": metrics(q, ff_profile, detector_profile),
        "geant4_truth_vs_F2_Thomson": metrics(q, ff_profile, truth_profile),
    }

    display_sigma_bins = 1.0
    displayed = {
        "XRD-preprocessing, detector image": gaussian_filter1d(
            normalize_area(q, detector_profile), display_sigma_bins
        ),
        "Geant4 truth, single Rayleigh": gaussian_filter1d(
            normalize_area(q, truth_profile), display_sigma_bins
        ),
        r"Keele $F^2/W$ × Thomson": normalize_area(q, ff_profile),
    }

    figure, axis = plt.subplots(figsize=(9.4, 6.0), constrained_layout=True)
    axis.plot(q, displayed["XRD-preprocessing, detector image"], color="#2563eb", lw=2.2)
    axis.plot(q, displayed["Geant4 truth, single Rayleigh"], color="#dc2626", lw=1.9)
    axis.plot(q, displayed[r"Keele $F^2/W$ × Thomson"], color="#16a34a", lw=2.0, ls="--")
    axis.set(
        title=(
            "50 mm breast tissue, Ag Kα1\n"
            f"thickness-corrected detector distance: {corrected_distance_m * 1e3:.1f} mm"
        ),
        xlabel=r"$q$ (nm$^{-1}$)",
        ylabel=r"Area-normalized coherent intensity",
        xlim=(args.q_min, args.q_max),
        ylim=(0.0, None),
    )
    axis.grid(alpha=0.25)
    axis.legend(
        [
            "XRD-preprocessing, detector image",
            "Geant4 truth, single Rayleigh",
            r"Keele $F^2/W$ × Thomson",
        ],
        frameon=False,
    )
    metric = comparisons["xrd_preprocessing_vs_geant4_truth"]
    axis.text(
        0.02,
        0.97,
        f"XRD vs G4 truth: r={metric['correlation']:.3f}, "
        f"rRMSE={100.0 * metric['relative_rmse']:.1f}%",
        transform=axis.transAxes,
        va="top",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)

    profile_path = args.output.with_name(f"{args.output.stem}_profiles.csv")
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "q_nm^-1",
                "xrd_preprocessing_single_rayleigh",
                "xrd_preprocessing_all_rayleigh",
                "geant4_truth_single_rayleigh",
                "keele_F2_over_W_times_Thomson",
            ]
        )
        writer.writerows(
            zip(
                q,
                detector_profile,
                all_rayleigh_profile,
                truth_profile,
                ff_profile,
                strict=True,
            )
        )

    metrics_path = args.output.with_name(f"{args.output.stem}_metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "comparison",
            "correlation",
            "relative_rmse",
            "reference_peak_q_nm^-1",
            "model_peak_q_nm^-1",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for comparison, values in comparisons.items():
            writer.writerow({"comparison": comparison, **values})

    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    summary = {
        "incident_photons": 100_000_000,
        **hit_counts,
        "sample_thickness_mm": args.sample_thickness_mm,
        "downstream_gap_mm": args.downstream_gap_mm,
        "front_face_reference_distance_mm": front_face_distance_m * 1e3,
        "thickness_corrected_distance_mm": corrected_distance_m * 1e3,
        "thickness_adjustment_applied": bool(
            corrected["thickness_adjustment_applied"]
        ),
        "thickness_adjustment_reliable": bool(
            corrected["thickness_adjustment_reliable"]
        ),
    }
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"corrected_distance_mm={corrected_distance_m * 1e3:.6f}")
    for name, values in comparisons.items():
        print(
            f"{name}: correlation={values['correlation']:.6f} "
            f"relative_rmse={values['relative_rmse']:.6f} "
            f"reference_peak={values['reference_peak_q_nm^-1']:.4f} "
            f"model_peak={values['model_peak_q_nm^-1']:.4f}"
        )
    for name, value in hit_counts.items():
        print(f"{name}={value}")
    print(f"image={args.output}")
    print(f"profiles={profile_path}")
    print(f"metrics={metrics_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
