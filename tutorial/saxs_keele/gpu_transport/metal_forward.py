"""Opt-in Metal forward backend for synthetic Keele inverse experiments.

Photoelectric/Compton attenuation follows material mass fractions and density.
MIFF Rayleigh attenuation follows density times the angular integral of S(q),
consistent with G4PenelopeRayleighModelMI's MolWeight cancellation.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from prepare_keele_ff import COMPONENTS, mixture_material
if __package__:
    from .benchmark_g4 import BUILD, G4_BINARY, make_macro
else:
    from benchmark_g4 import BUILD, G4_BINARY, make_macro


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHYSICS_22 = HERE / "results" / "multi_transport" / "g50_22p163_shell_joint.json"
PHYSICS_30 = HERE / "results" / "multi_transport" / "g50_30kev_shell_joint.json"
BASELINE_FRACTIONS = {"water": 0.385893, "fat": 0.510305, "collagen": 0.103802}
PURE_XS = ROOT / "data" / "pure_cross_sections.json"
OSC_PATTERN = re.compile(r"SAXS_COMPTON_OSC CustomMat (\d+) "
                         r"([\deE+.-]+) ([\deE+.-]+) ([\deE+.-]+)")


@lru_cache(maxsize=256)
def composition_compton_shells(water: float, fat: float,
                               collagen: float) -> tuple[tuple[float, float, float], ...]:
    """Export Geant4 Penelope-2008 oscillator table for one elemental mixture.

    One Geant4 photon initializes the physics list; image-only CSV scoring
    avoids ROOT and is discarded with the temporary directory. The Keele
    reference MIFF used in this export does not affect the Compton table.
    """
    fractions = {"water": water, "fat": fat, "collagen": collagen}
    density, elements = mixture_material(fractions)
    with tempfile.TemporaryDirectory(prefix="keele_trial_shells_") as temporary:
        stem = Path(temporary) / "oscillator_export"
        case = dict(thickness_mm="50", gap_mm="110", beam_radius_mm="0.1",
                    focus_from_entry_mm="320", density_scale="1", case="0")
        macro = make_macro(case, 1, stem)
        macro = re.sub(r"/det/setCustomMatDensity .*",
                       f"/det/setCustomMatDensity {density:.12f}", macro)
        ticks = {name: round(elements[name] * 1e12) for name in ("H", "C", "N")}
        ticks["O"] = int(1e12) - sum(ticks.values())
        for name, value in ticks.items():
            macro = re.sub(rf"/det/setCustomMat{name}massfract .*",
                           f"/det/setCustomMat{name}massfract {value / 1e12:.12f}", macro)
        macro_path = stem.with_suffix(".mac")
        macro_path.write_text(macro)
        environment = os.environ.copy()
        environment.update(SAXS_IMAGE_ONLY="1", SAXS_IMAGE_CSV="1",
                           SAXS_IMAGE_PIXELS="16", SAXS_IMAGE_PITCH_UM="100",
                           SAXS_DUMP_COMPTON_OSC="1")
        environment.pop("SAXS_DUMP_XS", None)
        run = subprocess.run([str(G4_BINARY), str(macro_path), "1"], cwd=BUILD,
                             env=environment, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"Geant4 oscillator export failed: {run.stdout[-2000:]} "
                               f"{run.stderr[-1000:]}")
        rows = {int(index): (float(strength), float(ion), float(hartree))
                for index, strength, ion, hartree in OSC_PATTERN.findall(run.stdout)}
    if not rows or sorted(rows) != list(range(len(rows))) or len(rows) > 64:
        raise RuntimeError("Invalid Geant4 Compton oscillator export")
    return tuple(rows[index] for index in range(len(rows)))


def pure_cross_sections() -> dict[str, np.ndarray]:
    source = json.loads(PURE_XS.read_text())["materials"]
    tables: dict[str, np.ndarray] = {}
    for name in COMPONENTS:
        rows = np.asarray(source[name], dtype=float)
        if rows.shape != (6, 4):
            raise ValueError(f"Expected six Geant4 cross-section rows for {name}")
        tables[name] = rows
    return tables


def rayleigh_integral(miff: np.ndarray, energy_kev: float) -> float:
    # Same 0.0001-rad theta grid used by G4PenelopeRayleighModelMI. The MIFF
    # ordinate is F/sqrt(W), so its square is S(q). Absolute normalization is
    # anchored by Geant4 cross sections of the g50 reference material.
    theta = np.arange(1, 31416, dtype=float) * 0.0001
    q_g4 = 2 * energy_kev / 510.99895 * np.sin(theta / 2)
    s = np.interp(q_g4, miff[:, 0], miff[:, 1] ** 2)
    return float(np.trapezoid((1 + np.cos(theta) ** 2) * np.sin(theta) * s, theta))


def physics_reference_path(energy_kev: float) -> Path:
    if np.isclose(energy_kev, 22.162917, atol=1e-5):
        return PHYSICS_22
    if np.isclose(energy_kev, 30.0, atol=1e-5):
        return PHYSICS_30
    raise ValueError(f"Metal backend is not calibrated at {energy_kev} keV")


def trial_physics(miff: np.ndarray, fractions: dict[str, float],
                  reference_path: Path = PHYSICS_22) -> dict:
    if set(fractions) != set(COMPONENTS) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError("Expected water/fat/collagen mass fractions summing to one")
    base = json.loads(reference_path.read_text())
    energies = np.asarray(base["energy_kev"], dtype=float)
    pure = pure_cross_sections()
    density, _ = mixture_material(fractions)
    base_density, _ = mixture_material(BASELINE_FRACTIONS)
    sample = np.asarray(base["sample_xs_mm_inv"], dtype=float)
    for channel in (0, 1):
        sample[channel] = density * sum(
            fractions[name] * pure[name][:, channel + 1]
            / float(COMPONENTS[name]["density"])
            for name in COMPONENTS
        )
    reference_miff = np.loadtxt(HERE.parent / "data" / "keele_breast_g50.dat")
    sample[2] = np.asarray(base["sample_xs_mm_inv"][2]) * density / base_density * np.array([
        rayleigh_integral(miff, energy) / rayleigh_integral(reference_miff, energy)
        for energy in energies
    ])
    base["sample_xs_mm_inv"] = sample.tolist()
    if "sample_compton_shells" in base:
        key = tuple(float(fractions[name]) for name in ("water", "fat", "collagen"))
        base["sample_compton_shells"] = [list(shell) for shell in composition_compton_shells(*key)]
        base["compton_shell_source"] = "Geant4 Penelope-2008 CustomMat; composition cached"
    base["trial_mass_fractions"] = fractions
    base["trial_density_g_cm3"] = density
    base["rayleigh_xs_method"] = "Geant4 g50 XS times density and S-angle-integral ratios"
    return base


def run_trial_metal(*, label: str, miff: np.ndarray,
                    transport_fractions: dict[str, float], manifest: dict,
                    photons: int, seeds: tuple[int, int], output_dir: Path) -> np.ndarray:
    reference_path = physics_reference_path(float(manifest["energy_kev"]))
    thickness = float(manifest["sample_thickness_mm"])
    gap = float(manifest["front_face_to_detector_mm"]) - thickness
    pixels = int(manifest["detector_pixels"])
    pitch = float(manifest["pixel_pitch_um"]) * 1e-3
    half = pixels * pitch / 2
    radius = float(manifest["source_diameter_um"]) * 5e-4
    entry_z = float(manifest["source_plane_z_mm"]) + 0.1
    focus = float(manifest["focus_point_z_mm"]) - entry_z
    physics = trial_physics(miff, transport_fractions, reference_path)
    physics["sample_lateral_mm"] = float(manifest.get("sample_lateral_mm", 120.0))
    with tempfile.TemporaryDirectory(prefix="keele_metal_trial_") as temporary:
        scratch = Path(temporary)
        ff = scratch / "trial_ff.dat"
        table = scratch / "trial_physics.json"
        raw = scratch / "image.raw"
        np.savetxt(ff, miff, fmt="%.12e")
        table.write_text(json.dumps(physics, separators=(",", ":")))
        command = [str(HERE / "multi_transport"), str(photons), str(thickness), str(gap),
                   str(half), str(pitch), str(radius), str(focus), "1", "1", "1", "1",
                   str(seeds[0]), str(table), str(ff), str(raw)]
        summary = json.loads(subprocess.check_output(command, text=True))
        if summary["interaction_cap_count"]:
            raise RuntimeError(f"Metal interaction cap reached: {summary}")
        image = np.fromfile(raw, dtype=np.uint32).reshape(pixels, pixels).astype(np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{label}.npz", image=image.astype(np.int32), miff=miff)
    (output_dir / f"{label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return image
