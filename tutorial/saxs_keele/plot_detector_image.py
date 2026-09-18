#!/usr/bin/env python3
"""Pixelize a Geant4 SAXS detector ROOT file and render photon-count maps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from scipy.ndimage import gaussian_filter1d
import uproot

HC_EV_M = 1.2398419843320026e-6
ELECTRON_REDUCED_COMPTON_NM = 3.8615926744e-4
SILICON_DENSITY_G_CM3 = 2.329
# NIST XCOM total mass attenuation coefficients around the Ag K-alpha lines.
SILICON_ATTENUATION = np.array(
    [(15.0, 10.34), (20.0, 4.464), (30.0, 1.436)], dtype=float
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--pixels", type=int, default=256)
    parser.add_argument("--pitch-um", type=float, default=200.0)
    parser.add_argument("--incident", type=int, default=100_000)
    parser.add_argument("--energy-kev", type=float, default=20.0)
    parser.add_argument("--distance-mm", type=float, default=100.0)
    parser.add_argument("--sample-thickness-mm", type=float, default=10.0)
    parser.add_argument("--sample-label", default="water")
    parser.add_argument("--sensor-thickness-um", type=float, default=300.0)
    parser.add_argument("--dead-layer-um", type=float, default=1.0)
    parser.add_argument("--threshold-kev", type=float, default=5.0)
    parser.add_argument("--charge-cloud-sigma-um", type=float, default=15.0)
    parser.add_argument("--detector-seed", type=int, default=20260915)
    parser.add_argument(
        "--ideal-detector",
        action="store_true",
        help="Register every primary photon crossing the active area",
    )
    parser.add_argument("--form-factor-file", type=Path)
    parser.add_argument("--radial-points", type=int, default=100)
    parser.add_argument("--q-min", type=float, default=1.0)
    parser.add_argument("--q-max", type=float)
    parser.add_argument(
        "--include-corners",
        action="store_true",
        help=(
            "Integrate to detector corners; bins above the edge limit have "
            "partial azimuthal coverage"
        ),
    )
    return parser.parse_args()


def histogram(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _, _ = np.histogram2d(y, x, bins=(edges, edges))
    return counts


def silicon_mass_attenuation(energy_kev: np.ndarray) -> np.ndarray:
    """Log-log interpolation of NIST XCOM mu/rho in cm2/g."""
    energy = np.clip(energy_kev, SILICON_ATTENUATION[0, 0], SILICON_ATTENUATION[-1, 0])
    return np.exp(
        np.interp(
            np.log(energy),
            np.log(SILICON_ATTENUATION[:, 0]),
            np.log(SILICON_ATTENUATION[:, 1]),
        )
    )


def load_hits(paths: list[Path]) -> dict[str, np.ndarray]:
    columns = [
        "e",
        "posx",
        "posy",
        "momx",
        "momy",
        "momz",
        "type",
        "trackID",
        "NRi",
        "NCi",
    ]
    chunks: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    for path in paths:
        with uproot.open(path) as root_file:
            arrays = root_file["part"].arrays(columns, library="np")
        for column in columns:
            chunks[column].append(arrays[column])
    return {column: np.concatenate(values) for column, values in chunks.items()}


def detector_response(
    hits: dict[str, np.ndarray], selected: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Si dead layer, active absorption, and charge-cloud blur."""
    if args.ideal_detector:
        indices = np.flatnonzero(selected)
        return (
            selected.copy(),
            hits["posx"][indices].copy(),
            hits["posy"][indices].copy(),
        )

    rng = np.random.default_rng(args.detector_seed)
    indices = np.flatnonzero(selected)
    energy = hits["e"][indices]
    cos_incidence = np.clip(hits["momz"][indices], 1.0e-6, 1.0)
    mu_cm_inv = silicon_mass_attenuation(energy) * SILICON_DENSITY_G_CM3
    dead_path_cm = args.dead_layer_um * 1.0e-4 / cos_incidence
    active_path_cm = args.sensor_thickness_um * 1.0e-4 / cos_incidence
    probability = np.exp(-mu_cm_inv * dead_path_cm) * (
        1.0 - np.exp(-mu_cm_inv * active_path_cm)
    )
    accepted = (energy >= args.threshold_kev) & (rng.random(indices.size) < probability)
    detected = np.zeros(selected.size, dtype=bool)
    detected[indices[accepted]] = True
    sigma_mm = args.charge_cloud_sigma_um / 1000.0
    detected_indices = indices[accepted]
    x = hits["posx"][detected_indices] + rng.normal(
        0.0, sigma_mm, detected_indices.size
    )
    y = hits["posy"][detected_indices] + rng.normal(
        0.0, sigma_mm, detected_indices.size
    )
    return detected, x, y


def form_factor_profile(path: Path, q_nm: np.ndarray, energy_kev: float) -> np.ndarray:
    table = np.loadtxt(path)
    table_q_nm = table[:, 0] / ELECTRON_REDUCED_COMPTON_NM
    intensity = table[:, 1] ** 2
    wavelength_nm = HC_EV_M * 1.0e9 / (energy_kev * 1000.0)
    sin_half_theta = np.clip(q_nm * wavelength_nm / (4.0 * math.pi), 0.0, 1.0)
    theta = 2.0 * np.arcsin(sin_half_theta)
    thomson_factor = 0.5 * (1.0 + np.cos(theta) ** 2)
    return np.interp(q_nm, table_q_nm, intensity) * thomson_factor


def main() -> None:
    args = parse_args()
    pitch_mm = args.pitch_um / 1000.0
    side_mm = args.pixels * pitch_mm
    half_mm = side_mm / 2.0
    edges = np.linspace(-half_mm, half_mm, args.pixels + 1)

    hits = load_hits(args.root_files)

    primary_gamma = (hits["type"] == 0) & (hits["trackID"] == 1) & (hits["momz"] > 0)
    in_active_area = (
        (hits["posx"] >= -half_mm)
        & (hits["posx"] < half_mm)
        & (hits["posy"] >= -half_mm)
        & (hits["posy"] < half_mm)
    )
    selected = primary_gamma & in_active_area
    coherent = selected & (hits["NRi"] == 1) & (hits["NCi"] == 0)
    multiple_rayleigh = selected & (hits["NRi"] > 1) & (hits["NCi"] == 0)
    compton = selected & (hits["NCi"] > 0)
    direct = selected & (hits["NRi"] == 0) & (hits["NCi"] == 0)

    detected, detected_x, detected_y = detector_response(hits, selected, args)
    detected_indices = np.flatnonzero(detected)
    detected_lookup = np.full(selected.size, -1, dtype=np.int64)
    detected_lookup[detected_indices] = np.arange(detected_indices.size)
    detected_coherent = detected & coherent
    detected_multiple_rayleigh = detected & multiple_rayleigh
    detected_compton = detected & compton
    detected_direct = detected & direct
    coherent_xy = detected_lookup[np.flatnonzero(detected_coherent)]
    all_image = histogram(detected_x, detected_y, edges)
    coherent_image = histogram(detected_x[coherent_xy], detected_y[coherent_xy], edges)

    wavelength_m = HC_EV_M / (args.energy_kev * 1000.0)
    # The requested distance is the air gap after the water. pyFAI requires
    # the distance from the scattering reference point, taken at water centre.
    integration_distance_mm = args.distance_mm + args.sample_thickness_mm / 2.0
    full_ring_radius_m = half_mm / 1000.0
    full_ring_angle = math.atan2(full_ring_radius_m, integration_distance_mm / 1000.0)
    full_ring_q_max = (
        4.0 * math.pi * math.sin(full_ring_angle / 2.0) / (wavelength_m * 1e9)
    )
    detector_radius_m = full_ring_radius_m * (
        math.sqrt(2.0) if args.include_corners else 1.0
    )
    detector_angle = math.atan2(detector_radius_m, integration_distance_mm / 1000.0)
    detector_q_max = (
        4.0 * math.pi * math.sin(detector_angle / 2.0) / (wavelength_m * 1e9)
    )
    q_max = args.q_max if args.q_max is not None else detector_q_max
    integrator = AzimuthalIntegrator(
        dist=integration_distance_mm / 1000.0,
        poni1=half_mm / 1000.0,
        poni2=half_mm / 1000.0,
        pixel1=pitch_mm / 1000.0,
        pixel2=pitch_mm / 1000.0,
        wavelength=wavelength_m,
    )
    ring_sum_kwargs = {
        "npt": args.radial_points,
        "unit": "q_nm^-1",
        "correctSolidAngle": False,
        "method": ("no", "histogram", "cython"),
        "radial_range": (args.q_min, q_max),
    }
    solid_angle_intensity_kwargs = {
        **ring_sum_kwargs,
        "correctSolidAngle": True,
    }
    all_ring_sum_profile = integrator.integrate1d(all_image, **ring_sum_kwargs)
    coherent_ring_sum_profile = integrator.integrate1d(
        coherent_image, **ring_sum_kwargs
    )
    all_solid_angle_profile = integrator.integrate1d(
        all_image, **solid_angle_intensity_kwargs
    )
    coherent_solid_angle_profile = integrator.integrate1d(
        coherent_image, **solid_angle_intensity_kwargs
    )

    cmap = plt.colormaps["magma"].copy()
    extent = (-half_mm, half_mm, -half_mm, half_mm)
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), constrained_layout=True)
    panels = (
        (all_image, "All primary photons", axes[0]),
        (coherent_image, "Single Rayleigh scattering", axes[1]),
    )
    for image, title, axis in panels:
        rendered = axis.imshow(
            np.log10(image + 1.0),
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
        )
        axis.set(
            title=f"{title}\n{int(image.sum()):,} detected",
            xlabel="x on detector (mm)",
            ylabel="y on detector (mm)",
            aspect="equal",
        )
        figure.colorbar(
            rendered, ax=axis, fraction=0.046, label=r"$\log_{10}(N+1)$ per pixel"
        )

    radial_q = coherent_solid_angle_profile.radial
    pyfai_shape = gaussian_filter1d(
        np.nan_to_num(coherent_solid_angle_profile.intensity), 1.0
    )
    if pyfai_shape.max() > 0:
        pyfai_shape /= pyfai_shape.max()
    coherent_energy = hits["e"][detected_coherent]
    coherent_theta = np.arccos(np.clip(hits["momz"][detected_coherent], -1.0, 1.0))
    coherent_wavelength_nm = HC_EV_M * 1.0e9 / (coherent_energy * 1000.0)
    truth_q = 4.0 * math.pi * np.sin(coherent_theta / 2.0) / coherent_wavelength_nm
    q_edges = np.linspace(args.q_min, q_max, args.radial_points + 1)
    truth_counts, _ = np.histogram(truth_q, bins=q_edges)
    # For constant-width q bins, dOmega/dq is proportional to q. Dividing by
    # bin-centre q converts accepted counts to a differential-solid-angle shape.
    truth_shape = gaussian_filter1d(truth_counts.astype(float) / radial_q, 1.0)
    if truth_shape.max() > 0:
        truth_shape /= truth_shape.max()
    axes[2].plot(
        radial_q, pyfai_shape, "o-", label="pyFAI: detector pixels", markersize=3
    )
    axes[2].plot(
        radial_q,
        truth_shape,
        label="Geant4 truth: solid-angle corrected",
        linewidth=1.8,
    )
    if args.form_factor_file:
        ff_shape = form_factor_profile(args.form_factor_file, radial_q, args.energy_kev)
        ff_shape /= ff_shape.max()
        axes[2].plot(
            radial_q, ff_shape, label=r"input $F^2(q)$ × Thomson", linewidth=1.8
        )
    axes[2].set(
        title=f"Radial shape ({args.radial_points} points)",
        xlabel=r"$q$ (nm$^{-1}$)",
        ylabel="Normalized Rayleigh intensity",
    )
    axes[2].set_ylim(bottom=0.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    if args.include_corners and q_max > full_ring_q_max:
        axes[2].axvline(full_ring_q_max, color="0.35", linestyle="--", linewidth=1.2)
        axes[2].text(
            full_ring_q_max,
            axes[2].get_ylim()[1] * 0.96,
            "full-ring limit",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="0.35",
        )

    figure.suptitle(
        f"Geant4 + Keele {args.sample_label} form factor\n"
        f"{args.energy_kev:g} keV, {args.incident:,} photons, "
        f"{args.sample_thickness_mm:g} mm {args.sample_label}, "
        f"{args.distance_mm:g} mm gap, {args.pixels}×{args.pixels} pixels, "
        f"{args.pitch_um:g} µm pitch; "
        + (
            "ideal 100% detector"
            if args.ideal_detector
            else f"{args.sensor_thickness_um:g} µm Si"
        ),
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    profile_path = args.output.with_name(
        f"{args.output.stem}_pyfai_{args.radial_points}_points.csv"
    )
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "q_nm^-1",
                "all_primary_sum_signal_correctSolidAngle_false",
                "single_rayleigh_sum_signal_correctSolidAngle_false",
                "all_primary_intensity_correctSolidAngle_true",
                "single_rayleigh_intensity_correctSolidAngle_true",
            ]
        )
        writer.writerows(
            zip(
                all_ring_sum_profile.radial,
                all_ring_sum_profile.sum_signal,
                coherent_ring_sum_profile.sum_signal,
                all_solid_angle_profile.intensity,
                coherent_solid_angle_profile.intensity,
                strict=True,
            )
        )

    summary = {
        "incident_photons": args.incident,
        "primary_photons_entering_active_area": int(np.count_nonzero(selected)),
        "sensor_detected_primary_photons": int(np.count_nonzero(detected)),
        "sensor_detected_direct_photons": int(np.count_nonzero(detected_direct)),
        "sensor_detected_rayleigh_photons": int(np.count_nonzero(detected_coherent)),
        "sensor_detected_multiple_rayleigh_photons": int(
            np.count_nonzero(detected_multiple_rayleigh)
        ),
        "sensor_detected_compton_photons": int(np.count_nonzero(detected_compton)),
        "not_detected_in_active_area": int(args.incident - np.count_nonzero(selected)),
        "sensor_detection_fraction": float(np.count_nonzero(detected) / args.incident),
        "rayleigh_sensor_detection_fraction": float(
            np.count_nonzero(detected_coherent) / args.incident
        ),
        "full_azimuth_q_limit_nm^-1": float(full_ring_q_max),
        "corner_q_limit_nm^-1": float(detector_q_max),
        "ring_sum_correct_solid_angle": False,
        "intensity_correct_solid_angle": True,
    }
    summary_path = args.summary or args.output.with_suffix(".csv")
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"image: {args.output}")
    print(f"summary: {summary_path}")
    print(f"pyFAI profile: {profile_path}")


if __name__ == "__main__":
    main()
