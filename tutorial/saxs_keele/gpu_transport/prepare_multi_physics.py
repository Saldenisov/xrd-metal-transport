"""Extract Geant4 attenuation and Penelope Compton angle law for Metal.

The Geant4 ROOT file lives only inside TemporaryDirectory. The persistent
artifact is a small JSON table, not event histories or a ROOT file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import uproot

from benchmark_g4 import BUILD, G4_BINARY, make_macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photons", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--material", choices=("g50", "water"), default="g50")
    parser.add_argument("--energy-kev", type=float, default=22.162917)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.photons < 1000:
        raise ValueError("Need enough photons to estimate Compton angle CDF")
    case = dict(thickness_mm="50", gap_mm="110", beam_radius_mm="0.1",
                focus_from_entry_mm="320", density_scale="1", case="0")
    with tempfile.TemporaryDirectory(prefix="keele_metal_physics_") as temporary:
        stem = Path(temporary) / "g50"
        macro = stem.with_suffix(".mac")
        macro_text = make_macro(case, args.photons, stem)
        if args.material == "water":
            macro_text = re.sub(r"/det/setCustomMatDensity .*",
                                "/det/setCustomMatDensity 1.000000000000", macro_text)
            for element, fraction in (("H", "0.111900000000"),
                                      ("C", "0.000000000000"),
                                      ("N", "0.000000000000"),
                                      ("O", "0.888100000000")):
                macro_text = re.sub(rf"/det/setCustomMat{element}massfract .*",
                                    f"/det/setCustomMat{element}massfract {fraction}",
                                    macro_text)
            macro_text = macro_text.replace("data/keele_breast_g50.dat", "data/keele_water.dat")
        macro_text = macro_text.replace("22.162917 keV", f"{args.energy_kev:.9f} keV")
        macro.write_text(macro_text)
        environment = os.environ.copy()
        environment.pop("SAXS_IMAGE_ONLY", None)
        environment["SAXS_DUMP_XS"] = "1"
        environment["SAXS_DUMP_COMPTON_OSC"] = "1"
        environment.pop("SAXS_DUMP_XS_EXTRA_KEV", None)
        default_energies = (10.0, 15.0, 20.0, 22.162917, 25.0, 30.0)
        if min(abs(args.energy_kev - value) for value in default_energies) > 1e-6:
            environment["SAXS_DUMP_XS_EXTRA_KEV"] = f"{args.energy_kev:.9f}"
        run = subprocess.run([str(G4_BINARY), str(macro), str(args.threads)], cwd=BUILD,
                             env=environment, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        if run.returncode:
            raise RuntimeError(run.stdout[-4000:])
        pattern = re.compile(r"SAXS_XS (CustomMat|Air) ([\d.]+) phot=([\deE+.-]+) "
                             r"compt=([\deE+.-]+) Rayl=([\deE+.-]+)")
        rows: dict[str, list[list[float]]] = {"CustomMat": [], "Air": []}
        for match in pattern.finditer(run.stdout):
            rows[match.group(1)].append([float(value) for value in match.groups()[1:]])
        for material in ("CustomMat", "Air"):
            sampled = sorted(rows[material], key=lambda row: row[0])
            if len(sampled) == 7:
                sampled = [row for row in sampled if abs(row[0] - 22.162917) > 1e-6]
            if (len(sampled) != 6 or
                any(right[0] <= left[0] for left, right in zip(sampled, sampled[1:])) or
                min(abs(row[0] - args.energy_kev) for row in sampled) > 1e-6):
                raise RuntimeError(f"Missing exact source-energy Geant4 XS for {material}")
            rows[material] = sampled
        oscillator_rows: dict[str, dict[int, list[float]]] = {"CustomMat": {}, "Air": {}}
        osc_pattern = re.compile(r"SAXS_COMPTON_OSC (CustomMat|Air) (\d+) "
                                 r"([\deE+.-]+) ([\deE+.-]+) ([\deE+.-]+)")
        for match in osc_pattern.finditer(run.stdout):
            material, index, strength, ion_kev, hartree = match.groups()
            oscillator_rows[material][int(index)] = [float(strength), float(ion_kev),
                                                     float(hartree)]
        shells = {}
        for material in ("CustomMat", "Air"):
            indexed = oscillator_rows[material]
            if not indexed or sorted(indexed) != list(range(len(indexed))) or len(indexed) > 64:
                raise RuntimeError(f"Invalid Geant4 Penelope Compton shell table for {material}")
            shells[material] = [indexed[i] for i in range(len(indexed))]
        counts = np.zeros(1024, dtype=np.int64)
        probe_angles: list[np.ndarray] = []
        probe_energies: list[np.ndarray] = []
        for batch in uproot.iterate(f"{stem.with_suffix('.root')}:scatt",
                                    ["processID", "theta", "e", "e_after"], library="np",
                                    step_size="100 MB"):
            selected = (batch["processID"] == 2) & (abs(batch["e"] - args.energy_kev) < 0.001)
            angles = np.deg2rad(batch["theta"][selected])
            counts += np.histogram(angles, bins=1024, range=(0, np.pi))[0]
            probe_angles.append(angles)
            probe_energies.append(batch["e_after"][selected])
        if counts.sum() < 10_000:
            raise RuntimeError(f"Only {counts.sum()} first-energy Compton events")
        # A half-count per angle bin prevents flat CDF intervals; its total
        # influence is below 0.2% for the default calibration sample.
        cumulative = np.r_[0.0, np.cumsum(counts + 0.5)]
        cumulative /= cumulative[-1]
        angles = np.concatenate(probe_angles)
        energies = np.concatenate(probe_energies)
        angle_edges = np.linspace(0, np.pi, 65)
        energy_edges = np.linspace(0, args.energy_kev * 1.001, 65)
        joint_angle_edges = np.linspace(0, np.pi, 33)
        joint_energy_edges = np.linspace(0, args.energy_kev * 1.001, 33)
        payload = {
            "source": f"Geant4 empenelopeMI, CustomMat {args.material}, {args.energy_kev} keV",
            "source_energy_kev": args.energy_kev,
            "sample_lateral_mm": 120.0,
            "photons": args.photons,
            "compton_events": int(counts.sum()),
            "energy_kev": [row[0] for row in rows["CustomMat"]],
            "sample_xs_mm_inv": [[row[j] for row in rows["CustomMat"]] for j in (1, 2, 3)],
            "air_xs_mm_inv": [[row[j] for row in rows["Air"]] for j in (1, 2, 3)],
            "compton_cdf": cumulative.tolist(),
            "sample_compton_shells": shells["CustomMat"],
            "air_compton_shells": shells["Air"],
            "compton_probe_reference": {
                "selection": "sample Compton events with incoming photon energy within 0.001 keV of source",
                "count": int(angles.size),
                "theta_mean_rad": float(angles.mean()),
                "theta_sd_rad": float(angles.std()),
                "e_after_mean_kev": float(energies.mean()),
                "e_after_sd_kev": float(energies.std()),
                "theta_hist_64": np.histogram(angles, bins=angle_edges)[0].tolist(),
                "e_after_hist_64": np.histogram(energies, bins=energy_edges)[0].tolist(),
                "theta_energy_hist_32x32": np.histogram2d(
                    angles, energies, bins=(joint_angle_edges, joint_energy_edges))[0]
                    .astype(int).tolist(),
            },
            "limits": "Compton CDF calibrated at incident energy and g50 composition; "
                      "shell/Doppler Compton available to Metal; no fluorescence or detector response",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "compton_events": int(counts.sum()),
                      "roots_retained": 0}))


if __name__ == "__main__":
    main()
