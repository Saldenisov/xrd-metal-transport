"""Compare Metal multi-collision transport to independent Geant4 histories.

The same finite 50 mm box, air gap, focused Ag K-alpha1 beam and ideal 1000²
detector are used. Geant4 writes CSV images only; no ROOT file is created.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, diags
from xrd_preprocessing.azimuthal import _integrator_from_dataframe

from benchmark_g4 import BUILD, DEFAULT_FF, G4_BINARY, HERE, make_macro
from geometry import SlabDetectorGeometry
from validate_water_100mm_cpu_metal import (CHANNELS, read_g4_channels,
                                              read_g4_image, profile_parity)
sys.path.insert(0, str(HERE.parent))
from metal_forward import trial_physics
from prepare_keele_ff import mixture_material
from check_trial_xs import trial_miff
from compare_breast_50mm_xrd_preprocessing import integrate_with_xrd_preprocessing


def integrate(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    record = integrate_with_xrd_preprocessing(
        image, pixel_um=100.0, wavelength_m=1.2398419843320026e-9 / 22.162917,
        front_face_distance_m=0.160, sample_thickness_mm=50.0,
        q_range=(1.0, 30.0), npt=256, correct_solid_angle=False,
    )
    return (np.asarray(record["q_range"], dtype=float),
            np.asarray(record["radial_profile_data"], dtype=float))


def pyfai_operator() -> tuple[np.ndarray, csr_matrix]:
    wavelength_nm = 1.2398419843320026 / 22.162917
    ai = _integrator_from_dataframe(100e-6, 500, 500, wavelength_nm * 10, 135.0)
    result = ai.integrate1d(np.zeros((1000, 1000), dtype=np.float32), 256,
                            unit="q_nm^-1", radial_range=(1.0, 30.0),
                            error_model="poisson", correctSolidAngle=False)
    engine = ai.engines[result.method].engine
    raw_weights = csr_matrix((engine.data, engine.indices, engine.indptr),
                             shape=(256, 1_000_000))
    normalization = np.asarray(result.count, dtype=float)
    if not np.allclose(np.asarray(raw_weights.sum(axis=1)).ravel(), normalization):
        raise AssertionError("CSR weights and pyFAI pixel counts differ")
    weights = diags(1 / normalization) @ raw_weights
    return np.asarray(result.radial), weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photons", type=int, default=2_000_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--fractions", type=float, nargs=3,
                        metavar=("WATER", "FAT", "COLLAGEN"))
    parser.add_argument("--residual-peak", action="store_true",
                        help="Add a hidden S(q) Gaussian near 6 nm^-1")
    parser.add_argument("--metal-physics", type=Path,
                        help="Prepared Geant4 cross sections and Penelope shell tables")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "multi_transport")
    args = parser.parse_args()
    if args.metal_physics is not None and (args.fractions is not None or args.residual_peak):
        parser.error("--metal-physics cannot be combined with an unprepared trial material")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="keele_multi_validate_") as temporary:
        stem = Path(temporary) / "g50"
        macro = stem.with_suffix(".mac")
        geometry = SlabDetectorGeometry(
            sample_thickness_mm=50.0, sample_lateral_mm=120.0,
            downstream_air_mm=110.0, detector_pixels=1000,
            pixel_pitch_mm=0.1, beam_radius_mm=0.1,
            focus_from_entry_mm=320.0,
        )
        case = dict(thickness_mm="50", gap_mm="110", beam_radius_mm="0.1",
                    focus_from_entry_mm="320", density_scale="1", case="0")
        macro_text = make_macro(case, args.photons, stem)
        physics_path = HERE / "results" / "multi_physics_g50.json"
        if args.metal_physics is not None:
            physics_path = args.metal_physics
        ff_path = DEFAULT_FF
        if args.fractions is not None or args.residual_peak:
            values = args.fractions or [0.385893, 0.510305, 0.103802]
            fractions = dict(zip(("water", "fat", "collagen"), values))
            if min(values) < 0 or not np.isclose(sum(values), 1):
                raise ValueError("Fractions must be nonnegative and sum to one")
            miff = trial_miff(fractions)
            if args.residual_peak:
                q_nm = miff[:, 0] / 3.8615926744e-4
                miff[:, 1] = np.sqrt(miff[:, 1] ** 2
                                       + 0.3 * np.exp(-0.5 * ((q_nm - 6.0) / 0.8) ** 2))
            ff_path = Path(temporary) / "trial_ff.dat"
            np.savetxt(ff_path, miff, fmt="%.12e")
            physics_path = Path(temporary) / "trial_physics.json"
            physics_path.write_text(json.dumps(trial_physics(miff, fractions)))
            density, elements = mixture_material(fractions)
            ticks = {name: round(elements[name] * 1e12) for name in ("H", "C", "N")}
            ticks["O"] = int(1e12) - sum(ticks.values())
            macro_text = re.sub(r"/det/setCustomMatDensity .*",
                                f"/det/setCustomMatDensity {density:.12f}", macro_text)
            for name, value in ticks.items():
                macro_text = re.sub(rf"/det/setCustomMat{name}massfract .*",
                                    f"/det/setCustomMat{name}massfract {value / 1e12:.12f}",
                                    macro_text)
            macro_text = macro_text.replace("data/keele_breast_g50.dat", str(ff_path))
        macro.write_text(macro_text)
        environment = os.environ.copy()
        environment["SAXS_IMAGE_ONLY"] = "1"
        environment["SAXS_IMAGE_CSV"] = "1"
        environment["SAXS_IMAGE_PIXELS"] = "1000"
        environment["SAXS_IMAGE_PITCH_UM"] = "100"
        environment.pop("SAXS_DUMP_XS", None)
        start = time.perf_counter()
        g4 = subprocess.run([str(G4_BINARY), str(macro), str(args.threads)],
                            cwd=BUILD, env=environment, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
        g4_seconds = time.perf_counter() - start
        if g4.returncode:
            raise RuntimeError(g4.stdout[-4000:])
        if stem.with_suffix(".root").exists():
            raise AssertionError("Geant4 unexpectedly created a ROOT file")
        reference = read_g4_image(stem.with_name(f"{stem.name}_h2_image.csv"))
        counts = read_g4_channels(stem.with_name(f"{stem.name}_h1_channels.csv"))
        g4_counts = dict(zip(CHANNELS, map(int, counts)))
        if reference.sum() != counts.sum():
            raise AssertionError("Geant4 image/channel counts differ")

        raw = Path(temporary) / "metal_image.raw"
        command = [str(HERE / "multi_transport"), str(args.photons),
                   *geometry.metal_arguments(), "1", "1", "1", "1", "42091",
                   str(physics_path), str(ff_path), str(raw)]
        start = time.perf_counter()
        metal = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
        metal_seconds = time.perf_counter() - start
        if metal.returncode:
            raise RuntimeError(metal.stderr)
        gpu_summary = json.loads(metal.stdout)
        shell_model = "Penelope-2008 shell/Doppler" in gpu_summary["physics_scope"]
        trial = np.fromfile(raw, dtype=np.uint32).reshape(1000, 1000)

    q_g4, i_g4 = integrate(reference)
    q_gpu, i_gpu = integrate(trial)
    if not np.allclose(q_g4, q_gpu):
        raise AssertionError("Geant4/Metal integration q axes differ")
    q_operator, weights = pyfai_operator()
    if not np.array_equal(q_g4, q_operator):
        raise AssertionError("XRD-preprocessing and CSR q axes differ")
    if not np.allclose(weights @ reference.ravel(), i_g4, rtol=1e-9, atol=1e-7):
        raise AssertionError("Geant4 XRD-preprocessing and CSR intensities differ")
    if not np.allclose(weights @ trial.ravel(), i_gpu, rtol=1e-9, atol=1e-7):
        raise AssertionError("Metal XRD-preprocessing and CSR intensities differ")
    parity = profile_parity(reference, trial, q_operator, weights)
    visible = (q_g4 >= 2) & (q_g4 <= 28) & np.isfinite(i_g4) & np.isfinite(i_gpu)
    ratio = np.divide(i_gpu, i_g4, out=np.full_like(i_g4, np.nan), where=i_g4 > 0)
    relative_rms = float(np.sqrt(np.mean((ratio[visible] - 1) ** 2)))
    relative_bias = float(np.mean(ratio[visible] - 1))
    expected = np.array([g4_counts["direct"], g4_counts["single_rayleigh"],
                         g4_counts["multiple_rayleigh"], g4_counts["compton"]])
    predicted = np.asarray(gpu_summary["classes"][:4])
    summary = {
        "photons": args.photons, "g4_threads": args.threads,
        "mass_fractions_water_fat_collagen": args.fractions,
        "added_unknown_residual_peak": args.residual_peak,
        "g4_seconds_wall": g4_seconds, "metal_seconds_wall": metal_seconds,
        "metal_seconds_transport_and_reduction": gpu_summary["seconds_total"],
        "metal_kernel_seconds": gpu_summary["seconds_gpu_kernel"],
        "wall_speedup": g4_seconds / metal_seconds,
        "transport_speedup_excluding_metal_startup": g4_seconds / gpu_summary["seconds_total"],
        "g4_counts": g4_counts, "metal_counts": gpu_summary["classes"],
        "channel_relative_difference": dict(zip(g4_counts, ((predicted / expected) - 1).tolist())),
        "g4_detector_total": int(reference.sum()),
        "metal_detector_total": int(trial.sum()),
        "radial_relative_rms_2_to_28_nm_inv": relative_rms,
        "radial_mean_relative_bias_2_to_28_nm_inv": relative_bias,
        "profile_parity": {key: value for key, value in parity.items()
                           if not isinstance(value, np.ndarray)},
        "roots_created": 0, "roots_retained": 0,
        "not_yet_geant4_equivalent": (["photoelectric fluorescence", "Rayleigh CDF sampling "
                                       "is approximate rather than Geant4 RITA", "detector response"]
                                      if shell_model else ["Compton shell/Doppler final state",
                                      "photoelectric fluorescence", "detector response"]),
    }
    (args.output_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "validation_profiles.npz", q_nm_inv=q_g4,
                        geant4=i_g4, metal=i_gpu)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, image, title in zip(axes[0], [reference, trial], ["Geant4", "Metal"]):
        ax.imshow(np.log1p(image), cmap="magma", extent=(-50, 50, -50, 50), origin="lower")
        ax.set(xlim=(-25, 25), ylim=(-25, 25), xlabel="x (mm)", ylabel="y (mm)", title=title)
    axes[1, 0].plot(q_g4, i_g4, label="Geant4", lw=1.5)
    axes[1, 0].plot(q_g4, i_gpu, label="Metal", lw=1.2)
    axes[1, 0].set(xlim=(1, 30), yscale="log", xlabel=r"$q$ (nm$^{-1}$)", ylabel="Intensity")
    axes[1, 0].legend()
    axes[1, 1].plot(q_g4, ratio - 1, lw=0.9)
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set(xlim=(1, 30), xlabel=r"$q$ (nm$^{-1}$)", ylabel="Metal / Geant4 - 1")
    fig.savefig(args.output_dir / "validation.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
