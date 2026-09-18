"""High-stat Geant4/Metal water comparison with the colleague's joblib.

Geant4 uses image-only CSV scoring. All intermediate CSV/raw files are kept
inside TemporaryDirectory; no ROOT files are created or retained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from benchmark_g4 import BUILD, G4_BINARY, HERE

sys.path.insert(0, str(HERE.parent))
from compare_colleague_water import (comparison_metrics,
                                     integrate_with_xrd_preprocessing,
                                     normalize_area, thickness_remap_q)


SOURCE = HERE.parent
WATER_FF = SOURCE / "data" / "keele_water.dat"
PHYSICS = HERE / "results" / "multi_transport" / "water_22p022_physics.json"


def geometry(frame) -> tuple[np.ndarray, tuple[float, float], float, float, float]:
    q = np.asarray(frame.iloc[0]["q"], dtype=float)
    step = float(np.median(np.diff(q)))
    q_range = (float(q[0] - step / 2), float(q[-1] + step / 2))
    wavelength = float(frame.iloc[0]["wavelength"])
    distance = float(frame.iloc[0]["dist"])
    theta = 2 * math.asin(q_range[1] * wavelength * 1e9 / (4 * math.pi))
    side_mm = math.sqrt(2) * distance * 1e3 * math.tan(theta)
    pixel_um = side_mm
    return q, q_range, wavelength, side_mm, pixel_um


def read_g4_image(path: Path, pixels: int = 1000) -> np.ndarray:
    with path.open(encoding="ascii") as stream:
        header = [next(stream).strip() for _ in range(10)]
    if header[0] != "#class tools::histo::h2d":
        raise ValueError(f"Not a Geant4 image CSV: {path}")
    values = np.loadtxt(path, delimiter=",", skiprows=10, usecols=(0,))
    if values.size != (pixels + 2) ** 2:
        raise ValueError(f"Unexpected Geant4 image shape: {values.size}")
    image = values.reshape(pixels + 2, pixels + 2)[1:-1, 1:-1]
    if not np.allclose(image, np.rint(image), atol=1e-9):
        raise ValueError("Non-integer Geant4 detector counts")
    return np.rint(image).astype(np.int64)


def cpu_image(*, thickness: int, photons: int, threads: int,
              pitch_um: float, scratch: Path) -> tuple[np.ndarray, float]:
    template = (SOURCE / f"water_colleague_{thickness}mm_10m.mac").read_text()
    stem = scratch / f"water_{thickness}mm_cpu"
    macro = re.sub(r"/run/setfilenamesave .*", f"/run/setfilenamesave {stem}", template)
    macro = re.sub(r"/run/beamOn \d+", f"/run/beamOn {photons}", macro)
    macro = re.sub(r"/run/printProgress \d+", "/run/printProgress 10000000", macro)
    macro = macro.replace("/run/verbose 1", "/run/verbose 0")
    path = stem.with_suffix(".mac")
    path.write_text(macro)
    environment = os.environ.copy()
    environment["SAXS_IMAGE_ONLY"] = "1"
    environment["SAXS_IMAGE_CSV"] = "1"
    environment["SAXS_IMAGE_PIXELS"] = "1000"
    environment["SAXS_IMAGE_PITCH_UM"] = f"{pitch_um:.12g}"
    environment.pop("SAXS_DUMP_XS", None)
    started = time.perf_counter()
    run = subprocess.run([str(G4_BINARY), str(path), str(threads)], cwd=BUILD,
                         env=environment, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(run.stdout[-5000:])
    if stem.with_suffix(".root").exists():
        raise AssertionError("Image-only Geant4 produced a ROOT file")
    image = read_g4_image(stem.with_name(f"{stem.name}_h2_image.csv"))
    return image, elapsed


def gpu_image(*, thickness: int, photons: int, side_mm: float,
              scratch: Path) -> tuple[np.ndarray, dict, float]:
    table = json.loads(PHYSICS.read_text())
    table["upstream_air_mm"] = 100.0 - thickness / 2.0
    config = scratch / f"water_{thickness}mm_physics.json"
    config.write_text(json.dumps(table, separators=(",", ":")))
    raw = scratch / f"water_{thickness}mm_gpu.raw"
    command = [str(HERE / "multi_transport"), str(photons), str(thickness),
               str(500 - thickness), str(side_mm / 2), str(side_mm / 1000),
               "0.05", "1000000000", "1", "1", "1", "1", str(95000 + thickness),
               str(config), str(WATER_FF), str(raw)]
    started = time.perf_counter()
    run = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=False)
    elapsed = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(run.stderr)
    summary = json.loads(run.stdout)
    if summary["interaction_cap_count"]:
        raise RuntimeError(f"Capped GPU histories: {summary}")
    return np.fromfile(raw, dtype=np.uint32).reshape(1000, 1000), summary, elapsed


def profile(image: np.ndarray, *, thickness: int, wavelength: float,
            pixel_um: float, q_range: tuple[float, float], npt: int):
    result = integrate_with_xrd_preprocessing(
        image, thickness_mm=thickness, base_distance_m=0.5,
        wavelength_m=wavelength, pixel_um=pixel_um,
        q_range=q_range, npt=npt,
    )
    return np.asarray(result["q_range"], dtype=float), np.asarray(
        result["radial_profile_data"], dtype=float), float(result["calculated_distance"])


def poisson_ring_parity(cpu: np.ndarray, gpu: np.ndarray, *, q_axis: np.ndarray,
                        distance_mm: float, wavelength_m: float,
                        pixel_um: float) -> dict[str, float | int]:
    """Independent equal-exposure MC samples: expected reduced chi² ≈ 1."""
    pixels = cpu.shape[0]
    row, col = np.indices(cpu.shape)
    radius_mm = np.hypot(col + .5 - pixels / 2, row + .5 - pixels / 2) * pixel_um * 1e-3
    theta = np.arctan(radius_mm / distance_mm)
    q_map = 4 * np.pi / (wavelength_m * 1e9) * np.sin(theta / 2)
    step = float(np.median(np.diff(q_axis)))
    edges = np.r_[q_axis - step / 2, q_axis[-1] + step / 2]
    counts_cpu = np.histogram(q_map, bins=edges, weights=cpu)[0]
    counts_gpu = np.histogram(q_map, bins=edges, weights=gpu)[0]
    selected = (q_axis >= 5) & (q_axis <= 30) & ((counts_cpu + counts_gpu) > 0)
    deviance = (counts_cpu[selected] - counts_gpu[selected]) ** 2 / (
        counts_cpu[selected] + counts_gpu[selected])
    return {"q_bins": int(np.count_nonzero(selected)),
            "reduced_chi2": float(np.mean(deviance)),
            "integrated_count_relative_bias": float(
                counts_gpu[selected].sum() / counts_cpu[selected].sum() - 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photons", type=int, default=50_000_000)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--colleague-file", type=Path, required=True,
                        help="Private colleague water reference (not included in this repository)")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "results" / "colleague_water_cpu_gpu_50m")
    parser.add_argument("--reuse", action="store_true", help="Use saved CPU/GPU images")
    args = parser.parse_args()
    if args.photons < 1:
        raise ValueError("Positive photon count required")
    frame = joblib.load(args.colleague_file)
    if [float(value) for value in frame["thickness"]] != [20.0, 30.0, 40.0]:
        raise ValueError("Unexpected colleague water thicknesses")
    q_received, q_range, wavelength, side_mm, pixel_um = geometry(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_summary = json.loads((args.output_dir / "summary.json").read_text()) if (
        args.reuse and (args.output_dir / "summary.json").exists()) else None
    records = []
    with tempfile.TemporaryDirectory(prefix="keele_colleague_water_") as temporary:
        scratch = Path(temporary)
        for _, row in frame.iterrows():
            thickness = int(float(row["thickness"]))
            archive = args.output_dir / f"water_{thickness}mm_{args.photons // 1_000_000}m.npz"
            if args.reuse and archive.exists():
                with np.load(archive) as payload:
                    cpu = np.asarray(payload["cpu_image"])
                    gpu = np.asarray(payload["gpu_image"])
                if previous_summary is None:
                    raise ValueError("--reuse requires the prior summary.json")
                old = previous_summary["profiles"][str(thickness)]
                cpu_seconds = old["geant4_wall_seconds"]
                gpu_seconds = old["metal_wall_seconds"]
                gpu_summary = {"classes": old["metal_classes"]}
            else:
                print(f"Geant4 {thickness} mm: {args.photons:,} photons", flush=True)
                cpu, cpu_seconds = cpu_image(
                    thickness=thickness, photons=args.photons, threads=args.threads,
                    pitch_um=pixel_um, scratch=scratch,
                )
                print(f"Metal {thickness} mm: {args.photons:,} photons", flush=True)
                gpu, gpu_summary, gpu_seconds = gpu_image(
                    thickness=thickness, photons=args.photons, side_mm=side_mm,
                    scratch=scratch,
                )
                np.savez_compressed(archive, cpu_image=cpu.astype(np.int32),
                                    gpu_image=gpu.astype(np.int32))
            q_cpu, i_cpu, corrected_cpu = profile(
                cpu, thickness=thickness, wavelength=wavelength, pixel_um=pixel_um,
                q_range=q_range, npt=len(q_received))
            q_gpu, i_gpu, corrected_gpu = profile(
                gpu, thickness=thickness, wavelength=wavelength, pixel_um=pixel_um,
                q_range=q_range, npt=len(q_received))
            if not np.allclose(q_cpu, q_gpu):
                raise AssertionError("CPU/GPU q axes differ")
            colleague = np.interp(q_cpu, np.asarray(row["q"], dtype=float),
                                  np.asarray(row["I"], dtype=float))
            adjusted_q = thickness_remap_q(np.asarray(row["q"], dtype=float),
                                            wavelength, 0.5, corrected_cpu)
            colleague_remapped = np.interp(q_cpu, adjusted_q,
                                           np.asarray(row["I"], dtype=float))
            np.savez_compressed(args.output_dir / f"water_{thickness}mm_profiles.npz",
                                q_nm_inv=q_cpu, colleague=colleague,
                                colleague_thickness_remapped=colleague_remapped,
                                geant4=i_cpu, metal=i_gpu)
            records.append({
                "thickness_mm": thickness, "q": q_cpu,
                "colleague": colleague, "colleague_remapped": colleague_remapped,
                "cpu": i_cpu, "gpu": i_gpu,
                "g4_seconds": cpu_seconds, "metal_seconds": gpu_seconds,
                "g4_detector_hits": int(cpu.sum()), "metal_detector_hits": int(gpu.sum()),
                "gpu_classes": gpu_summary.get("classes"),
                "adjusted_distance_mm": corrected_cpu * 1000,
                "poisson_ring_parity": poisson_ring_parity(
                    cpu, gpu, q_axis=q_cpu, distance_mm=corrected_cpu * 1000,
                    wavelength_m=wavelength, pixel_um=pixel_um),
            })
            print(json.dumps({"thickness_mm": thickness, "g4_seconds": cpu_seconds,
                              "metal_seconds": gpu_seconds,
                              "g4_hits": int(cpu.sum()), "metal_hits": int(gpu.sum()),
                              "adjusted_distance_mm": corrected_cpu * 1000}), flush=True)

    # Unknown incident flux/detector normalization: fit ONE common amplitude
    # across all three thicknesses, not a separate scale for each profile.
    visible = [(r["q"] >= 5) & (r["q"] <= 30) for r in records]
    reference = np.concatenate([r["colleague"][mask] for r, mask in zip(records, visible)])
    scales = {}
    for model in ("cpu", "gpu"):
        prediction = np.concatenate([r[model][mask] for r, mask in zip(records, visible)])
        scales[model] = float(reference @ prediction / (prediction @ prediction))
    metrics = {}
    for r in records:
        t = r["thickness_mm"]
        metrics[str(t)] = {
            "adjusted_distance_mm": r["adjusted_distance_mm"],
            "geant4_wall_seconds": r["g4_seconds"],
            "metal_wall_seconds": r["metal_seconds"],
            "geant4_detector_hits": r["g4_detector_hits"],
            "metal_detector_hits": r["metal_detector_hits"],
            "metal_detector_relative_difference": r["metal_detector_hits"] / r["g4_detector_hits"] - 1,
            "metal_classes": r["gpu_classes"],
            "poisson_ring_parity_5_to_30_nm_inv": r["poisson_ring_parity"],
        }
        mask = (r["q"] >= 5) & (r["q"] <= 30)
        for model in ("cpu", "gpu"):
            c = comparison_metrics(r["q"], r["colleague"], r[model])
            c["single_scale_relative_rmse"] = float(
                np.linalg.norm(scales[model] * r[model][mask] - r["colleague"][mask])
                / np.linalg.norm(r["colleague"][mask]))
            metrics[str(t)][f"{model}_vs_colleague"] = c
        metrics[str(t)]["gpu_vs_cpu"] = comparison_metrics(r["q"], r["cpu"], r["gpu"])
        metrics[str(t)]["cpu_vs_colleague_thickness_remapped"] = comparison_metrics(
            r["q"], r["colleague_remapped"], r["cpu"])
    near_axis = {name: [float(np.mean(r[name][r["q"] < .2])) for r in records]
                 for name in ("colleague", "cpu", "gpu")}
    summary = {
        "colleague_source": "user-supplied joblib (not redistributed)",
        "colleague_runs": frame["run_id"].tolist(),
        "wavelength_m": wavelength, "energy_kev": 1.2398419843320026e-9 / wavelength,
        "source_to_sample_air_mm": [90, 85, 80],
        "inferred_square_side_mm": side_mm, "inferred_pixel_um": pixel_um,
        "geometry_assumption": "1000x1000 pixels and detector edge inferred from colleague q max; "
                               "joblib does not specify detector pixels, mask or incident flux",
        "incident_photons_per_simulation": args.photons,
        "common_scale_colleague_per_geant4": scales["cpu"],
        "common_scale_colleague_per_metal": scales["gpu"],
        "near_axis_intensity_ratios_20_30_40_mm": {
            name: [value / values[0] for value in values]
            for name, values in near_axis.items()
        },
        "profiles": metrics, "roots_created": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    for ax, r in zip(axes, records):
        q = r["q"]
        mask = (q >= 5) & (q <= 30)
        ax.plot(q[mask], gaussian_filter1d(normalize_area(q, r["colleague"]), 2)[mask],
                color="black", lw=1.6, label="colleague")
        ax.plot(q[mask], gaussian_filter1d(normalize_area(q, r["cpu"]), 2)[mask],
                color="#2563eb", lw=1.2, label="Geant4 CPU")
        ax.plot(q[mask], gaussian_filter1d(normalize_area(q, r["gpu"]), 2)[mask],
                color="#dc2626", lw=1.1, label="Metal GPU")
        ax.set(xlim=(5, 30), ylabel="area-normalized intensity",
               title=f"Water {r['thickness_mm']} mm; scattering-plane distance {r['adjusted_distance_mm']:g} mm")
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel(r"$q$ (nm$^{-1}$)")
    fig.savefig(args.output_dir / "colleague_water_cpu_gpu.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    for ax, r in zip(axes, records):
        q = r["q"]
        mask = (q >= 5) & (q <= 30)
        observed = gaussian_filter1d(normalize_area(q, r["colleague"]), 3)
        for name, color, label in (("cpu", "#2563eb", "Geant4 CPU"),
                                   ("gpu", "#dc2626", "Metal GPU")):
            predicted = gaussian_filter1d(normalize_area(q, r[name]), 3)
            difference = 100 * (predicted / observed - 1)
            ax.plot(q[mask], difference[mask], color=color, lw=1.2, label=label)
        ax.axhline(0, color="black", lw=.8)
        ax.set(xlim=(5, 30), ylim=(-12, 12), ylabel="difference (%)",
               title=f"Water {r['thickness_mm']} mm, area-normalized shape")
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel(r"$q$ (nm$^{-1}$)")
    fig.savefig(args.output_dir / "colleague_water_shape_residuals.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"),
                      "figure": str(args.output_dir / "colleague_water_cpu_gpu.png"),
                      "residual_figure": str(args.output_dir / "colleague_water_shape_residuals.png"),
                      "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
