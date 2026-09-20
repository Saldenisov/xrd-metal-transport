"""Shared runner and analysis for the public X-ray diffraction example suite."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from xrd_preprocessing.azimuthal import _integrator_from_dataframe

ROOT = Path(__file__).resolve().parents[2]
SAXS = ROOT / "tutorial" / "saxs_keele"
GPU = SAXS / "gpu_transport"
BUILD = ROOT / "build" / "saxs_keele"
sys.path[:0] = [str(GPU), str(SAXS)]

from benchmark_g4 import G4_BINARY, make_macro  # noqa: E402
from check_trial_xs import trial_miff  # noqa: E402
from geometry import SlabDetectorGeometry  # noqa: E402
from metal_forward import PHYSICS_22  # noqa: E402
from prepare_keele_ff import mixture_material  # noqa: E402
from profile_integration import guarded_radial_spec, integrate_image  # noqa: E402
from validate_water_100mm_cpu_metal import (  # noqa: E402
    CHANNELS,
    profile_parity,
    read_g4_channels,
    read_g4_image,
)

ENERGY_KEV = 22.162917
MEAN_DETECTOR_DISTANCE_MM = 150.0
PIXELS = 1000
PITCH_MM = 0.1
RADIAL_POINTS = 256
Q_RANGE = (1.0, 30.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geant4_seeds(case_index: int) -> list[int]:
    return [550_071 + case_index * 37, 770_053 + case_index * 41]


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text())["cases"]
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Case names must be unique")
    for case in cases:
        fractions = case["fractions"]
        if set(fractions) != {"water", "fat", "collagen"}:
            raise ValueError(f"Invalid components in {case['name']}")
        if min(fractions.values()) < 0 or not np.isclose(sum(fractions.values()), 1):
            raise ValueError(f"Fractions do not sum to one in {case['name']}")
    return cases


def case_geometry(case: dict) -> SlabDetectorGeometry:
    thickness = float(case["thickness_mm"])
    return SlabDetectorGeometry(
        sample_thickness_mm=thickness,
        sample_lateral_mm=120.0,
        downstream_air_mm=MEAN_DETECTOR_DISTANCE_MM - thickness / 2,
        detector_pixels=PIXELS,
        pixel_pitch_mm=PITCH_MM,
        beam_radius_mm=0.1,
        focus_from_entry_mm=320.0,
    )


def geometry_manifest(geometry: SlabDetectorGeometry) -> dict:
    return geometry.manifest(
        energy_kev=ENERGY_KEV,
        q_range_nm_inv=Q_RANGE,
        radial_points=RADIAL_POINTS,
    )


def material_macro(case: dict, geometry: SlabDetectorGeometry, photons: int,
                   output: Path, form_factor: Path, case_index: int) -> str:
    setup = {
        "thickness_mm": str(geometry.sample_thickness_mm),
        "gap_mm": str(geometry.downstream_air_mm),
        "beam_radius_mm": str(geometry.beam_radius_mm),
        "focus_from_entry_mm": str(geometry.focus_from_entry_mm),
        "density_scale": "1",
        "case": str(case_index),
    }
    macro = make_macro(setup, photons, output)
    density, elements = mixture_material(case["fractions"])
    macro = re.sub(
        r"/det/setCustomMatDensity .*",
        f"/det/setCustomMatDensity {density:.12f}",
        macro,
    )
    ticks = {name: round(elements[name] * 1e12) for name in ("H", "C", "N")}
    ticks["O"] = int(1e12) - sum(ticks.values())
    for element, value in ticks.items():
        macro = re.sub(
            rf"/det/setCustomMat{element}massfract .*",
            f"/det/setCustomMat{element}massfract {value / 1e12:.12f}",
            macro,
        )
    return macro.replace("data/keele_breast_g50.dat", str(form_factor))


def physics_from_geant4(log: str, geometry: SlabDetectorGeometry) -> dict:
    xs_pattern = re.compile(
        r"SAXS_XS (CustomMat|Air) ([\d.]+) phot=([\deE+.-]+) "
        r"compt=([\deE+.-]+) Rayl=([\deE+.-]+)"
    )
    rows: dict[str, list[list[float]]] = {"CustomMat": [], "Air": []}
    for match in xs_pattern.finditer(log):
        rows[match.group(1)].append([float(value) for value in match.groups()[1:]])
    for material in rows:
        rows[material].sort(key=lambda row: row[0])
        if len(rows[material]) != 6 or not any(
            np.isclose(row[0], ENERGY_KEV, atol=1e-6) for row in rows[material]
        ):
            raise RuntimeError(f"Missing six exact Geant4 XS rows for {material}")

    osc_pattern = re.compile(
        r"SAXS_COMPTON_OSC (CustomMat|Air) (\d+) "
        r"([\deE+.-]+) ([\deE+.-]+) ([\deE+.-]+)"
    )
    oscillators: dict[str, dict[int, list[float]]] = {"CustomMat": {}, "Air": {}}
    for match in osc_pattern.finditer(log):
        material, index, strength, ionization, hartree = match.groups()
        oscillators[material][int(index)] = [
            float(strength), float(ionization), float(hartree)
        ]
    shells = {}
    for material, indexed in oscillators.items():
        if not indexed or sorted(indexed) != list(range(len(indexed))) or len(indexed) > 64:
            raise RuntimeError(f"Invalid Geant4 Penelope shells for {material}")
        shells[material] = [indexed[index] for index in range(len(indexed))]

    physics = json.loads(PHYSICS_22.read_text())
    physics.update(
        source="Exact tables exported by the matching Geant4 suite run",
        source_energy_kev=ENERGY_KEV,
        upstream_air_mm=geometry.upstream_air_mm,
        sample_lateral_mm=geometry.sample_lateral_mm,
        energy_kev=[row[0] for row in rows["CustomMat"]],
        sample_xs_mm_inv=[[row[column] for row in rows["CustomMat"]]
                          for column in (1, 2, 3)],
        air_xs_mm_inv=[[row[column] for row in rows["Air"]]
                       for column in (1, 2, 3)],
        sample_compton_shells=shells["CustomMat"],
        air_compton_shells=shells["Air"],
    )
    return physics


def pyfai_operator(manifest: dict) -> tuple[np.ndarray, csr_matrix]:
    wavelength_nm = 1.2398419843320026 / float(manifest["energy_kev"])
    pixels = int(manifest["detector_pixels"])
    pitch_m = float(manifest["pixel_pitch_um"]) * 1e-6
    distance_mm = float(manifest["mean_scattering_plane_to_detector_mm"])
    integrator = _integrator_from_dataframe(
        pitch_m, pixels / 2, pixels / 2, wavelength_nm * 10, distance_mm
    )
    points = int(manifest["radial_points"])
    guarded_points, guarded_range = guarded_radial_spec(
        points, tuple(manifest["q_range_nm_inv"])
    )
    result = integrator.integrate1d(
        np.zeros((pixels, pixels), dtype=np.float32),
        guarded_points,
        unit="q_nm^-1",
        radial_range=guarded_range,
        error_model="poisson",
        correctSolidAngle=False,
    )
    engine = integrator.engines[result.method].engine
    guarded_weights = csr_matrix(
        (engine.data, engine.indices, engine.indptr),
        shape=(guarded_points, pixels * pixels),
    )
    return np.asarray(result.radial)[1:-1], guarded_weights[1:-1]


def run_geant4(case: dict, geometry: SlabDetectorGeometry, photons: int,
                threads: int, form_factor: Path, scratch: Path,
                case_index: int) -> tuple[np.ndarray, np.ndarray, dict, float]:
    stem = scratch / "geant4"
    macro = stem.with_suffix(".mac")
    macro.write_text(material_macro(
        case, geometry, photons, stem, form_factor, case_index,
    ))
    environment = os.environ.copy()
    environment.update(
        SAXS_IMAGE_ONLY="1",
        SAXS_IMAGE_CSV="1",
        SAXS_IMAGE_PIXELS=str(PIXELS),
        SAXS_IMAGE_PITCH_UM=str(PITCH_MM * 1000),
        SAXS_DUMP_XS="1",
        SAXS_DUMP_COMPTON_OSC="1",
    )
    started = time.perf_counter()
    process = subprocess.run(
        [str(G4_BINARY), str(macro), str(threads)],
        cwd=BUILD,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    if process.returncode or "command not found" in process.stdout.lower():
        raise RuntimeError(process.stdout[-6000:])
    image = read_g4_image(stem.with_name("geant4_h2_image.csv"))
    channels = read_g4_channels(stem.with_name("geant4_h1_channels.csv"))
    if int(image.sum()) != int(channels.sum()):
        raise AssertionError("Geant4 image and channel counts differ")
    return image, channels, physics_from_geant4(process.stdout, geometry), elapsed


def run_metal(geometry: SlabDetectorGeometry, photons: int, seed: int,
              physics: dict, form_factor: Path, scratch: Path,
              ) -> tuple[np.ndarray, dict, float]:
    physics_path = scratch / "physics.json"
    raw = scratch / "metal.raw"
    physics_path.write_text(json.dumps(physics, separators=(",", ":")))
    command = [
        str(GPU / "multi_transport"), str(photons), *geometry.metal_arguments(),
        "1", "1", "1", "1", str(seed), str(physics_path), str(form_factor), str(raw),
    ]
    started = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(process.stderr[-4000:])
    summary = json.loads(process.stdout)
    if summary["interaction_cap_count"]:
        raise RuntimeError(f"Metal interaction cap reached: {summary}")
    image = np.fromfile(raw, dtype=np.uint32).reshape(PIXELS, PIXELS).astype(np.int64)
    if int(image.sum()) != int(sum(summary["classes"][:4])):
        raise AssertionError("Metal image and channel counts differ")
    return image, summary, elapsed


def _plot_case(output: Path, case: dict, geometry: SlabDetectorGeometry,
               photons: int, q: np.ndarray, geant4_profile: np.ndarray,
               metal_profile: np.ndarray, bin_z: np.ndarray,
               geant4_image: np.ndarray, metal_image: np.ndarray,
               channel_relative: dict[str, float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    extent = (-50, 50, -50, 50)
    maximum = max(np.percentile(np.log1p(geant4_image), 99.99),
                  np.percentile(np.log1p(metal_image), 99.99))
    for axis, image, title in zip(
        axes[0], (geant4_image, metal_image), ("Geant4 CPU", "Metal GPU")
    ):
        axis.imshow(np.log1p(image), origin="lower", extent=extent, cmap="magma",
                    vmin=0, vmax=maximum)
        axis.set(xlim=(-25, 25), ylim=(-25, 25), xlabel="x (mm)", ylabel="y (mm)",
                 title=f"{title}: log(1 + counts)")
    axes[1, 0].plot(q, geant4_profile / photons, label="Geant4", lw=1.5)
    axes[1, 0].plot(q, metal_profile / photons, label="Metal", lw=1.0)
    axes[1, 0].set(xlim=Q_RANGE, yscale="log", xlabel=r"$q$ (nm$^{-1}$)",
                   ylabel="Azimuthal sum / incident photon")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(q, bin_z, lw=0.9, color="#315c8a")
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].axhline(3, color="gray", lw=0.7, ls="--")
    axes[1, 1].axhline(-3, color="gray", lw=0.7, ls="--")
    axes[1, 1].set(xlim=Q_RANGE, xlabel=r"$q$ (nm$^{-1}$)",
                   ylabel=r"Metal $-$ Geant4 / $\sigma$")
    fractions = case["fractions"]
    figure_title = (
        f"{case['name']}: {geometry.sample_thickness_mm:g} mm; "
        f"water/fat/collagen = {fractions['water']:.3g}/"
        f"{fractions['fat']:.3g}/{fractions['collagen']:.3g}; "
        f"{photons:,} photons per backend"
    )
    fig.suptitle(figure_title)
    fig.savefig(output / "detector_comparison.png", dpi=160)
    plt.close(fig)

    with (output / "channel_comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("channel", "relative_difference_metal_over_geant4_minus_1"))
        writer.writerows(channel_relative.items())


def run_case(case: dict, case_index: int, photons: int, threads: int,
             output_root: Path, keep_arrays: bool = False, force: bool = False) -> dict:
    output = output_root / case["name"]
    summary_path = output / "summary.json"
    if summary_path.exists() and not force:
        saved = json.loads(summary_path.read_text())
        if saved.get("incident_photons_per_backend") == photons:
            return saved
        raise ValueError(f"Existing {case['name']} result uses another photon count")
    output.mkdir(parents=True, exist_ok=True)
    geometry = case_geometry(case)
    manifest = geometry_manifest(geometry)
    seed = 910_019 + case_index * 1_000_003
    input_record = {
        **case,
        "incident_photons_per_backend": photons,
        "geant4_threads": threads,
        "geant4_seeds": geant4_seeds(case_index),
        "metal_seed": seed,
        "geometry": manifest,
        "source_energy_kev": ENERGY_KEV,
        "mean_plane_detector_distance_mm": MEAN_DETECTOR_DISTANCE_MM,
        "component_table_sha256": sha256_file(ROOT / "data" / "xrd_components.txt"),
    }

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"xrd_suite_{case['name']}_") as temporary:
        scratch = Path(temporary)
        form_factor = scratch / "form_factor.dat"
        np.savetxt(form_factor, trial_miff(case["fractions"]), fmt="%.12e")
        input_record["generated_form_factor_sha256"] = sha256_file(form_factor)
        (output / "input.json").write_text(json.dumps(input_record, indent=2) + "\n")
        g4_image, g4_channels, physics, g4_seconds = run_geant4(
            case, geometry, photons, threads, form_factor, scratch, case_index,
        )
        metal_image, metal_summary, metal_seconds = run_metal(
            geometry, photons, seed, physics, form_factor, scratch,
        )

    q_g4, profile_g4, distance_g4 = integrate_image(g4_image, manifest)
    q_metal, profile_metal, distance_metal = integrate_image(metal_image, manifest)
    q_operator, weights = pyfai_operator(manifest)
    if not (np.array_equal(q_g4, q_metal) and np.array_equal(q_g4, q_operator)):
        raise AssertionError("Geant4, Metal and pyFAI q grids differ")
    if not np.allclose(weights @ g4_image.ravel(), profile_g4, rtol=1e-9, atol=1e-7):
        raise AssertionError("Geant4 profile differs from the saved pyFAI operator")
    if not np.allclose(weights @ metal_image.ravel(), profile_metal, rtol=1e-9, atol=1e-7):
        raise AssertionError("Metal profile differs from the saved pyFAI operator")
    parity = profile_parity(g4_image, metal_image, q_operator, weights)

    g4_counts = dict(zip(CHANNELS, map(int, g4_channels)))
    metal_counts = dict(zip(CHANNELS, map(int, metal_summary["classes"][:4])))
    channel_relative = {
        name: metal_counts[name] / g4_counts[name] - 1 for name in CHANNELS
    }
    channel_z = {
        name: (metal_counts[name] - g4_counts[name])
        / np.sqrt(metal_counts[name] + g4_counts[name]) for name in CHANNELS
    }
    normalized_g4 = profile_g4 / profile_g4.sum()
    normalized_metal = profile_metal / profile_metal.sum()
    visible = (q_g4 >= 2) & (q_g4 <= 28)
    shape_l2 = float(
        np.linalg.norm(normalized_metal[visible] - normalized_g4[visible])
        / np.linalg.norm(normalized_g4[visible])
    )
    profile_correlation = float(np.corrcoef(
        normalized_g4[visible], normalized_metal[visible]
    )[0, 1])
    summary = {
        "case": case["name"],
        "group": case["group"],
        "model": "X-ray diffraction; finite homogeneous sample and ideal detector",
        "mass_fractions_water_fat_collagen": case["fractions"],
        "geometry": manifest,
        "incident_photons_per_backend": photons,
        "geant4_seconds_wall": g4_seconds,
        "metal_seconds_wall": metal_seconds,
        "wall_speedup": g4_seconds / metal_seconds,
        "geant4_counts": g4_counts,
        "metal_counts": metal_counts,
        "channel_relative_difference": channel_relative,
        "channel_poisson_z": channel_z,
        "profile": {
            **{key: value for key, value in parity.items()
               if not isinstance(value, np.ndarray)},
            "normalized_shape_relative_l2_2_to_28_nm_inv": shape_l2,
            "normalized_shape_correlation_2_to_28_nm_inv": profile_correlation,
        },
        "metal_device": metal_summary["metal_device_name"],
        "metal_seed": seed,
        "effective_distance_mm": distance_g4,
        "same_effective_distance": distance_g4 == distance_metal,
        "roots_retained": 0,
        "detector_arrays_retained": bool(keep_arrays),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "ideal entrance-counting detector",
            "homogeneous finite rectangular sample",
            "Metal Rayleigh CDF differs numerically from Geant4 RITA",
            "no electron transport or fluorescence in Metal",
        ],
    }
    with (output / "profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("q_nm_inv", "geant4_sum", "metal_sum", "bin_z"))
        writer.writerows(zip(q_g4, profile_g4, profile_metal, parity["bin_z"]))
    _plot_case(
        output, case, geometry, photons, q_g4, profile_g4, profile_metal,
        parity["bin_z"], g4_image, metal_image, channel_relative,
    )
    if keep_arrays:
        np.savez_compressed(output / "detector_arrays.npz",
                            geant4=g4_image.astype(np.int32),
                            metal=metal_image.astype(np.int32))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
