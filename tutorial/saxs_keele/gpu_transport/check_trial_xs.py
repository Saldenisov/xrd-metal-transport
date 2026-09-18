"""Audit trial Metal attenuation tables against exact Geant4 MIFF cross sections."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from benchmark_g4 import BUILD, G4_BINARY, HERE, make_macro
sys.path.insert(0, str(HERE.parent))
from metal_forward import ROOT, trial_physics
from prepare_keele_ff import (mixture_material, mixture_points, read_components,
                              read_reference_miff, stitch_reference_tails,
                              ELECTRON_REDUCED_COMPTON_NM)


def trial_miff(fractions: dict[str, float]) -> np.ndarray:
    measured = read_components(ROOT / "data" / "xrd_components.txt")
    led = os.environ.get("G4LEDATA")
    if led is None:
        raise RuntimeError("Set G4LEDATA before preparing a trial MIFF")
    reference = read_reference_miff(Path(led) / "penelope" / "rayleigh" / "MIFF")
    curves = {name: stitch_reference_tails(measured[name], reference[name])
              for name in fractions}
    points = np.asarray(mixture_points(curves, fractions))
    return np.column_stack((points[:, 0] * ELECTRON_REDUCED_COMPTON_NM,
                            np.sqrt(points[:, 1])))


def g4_xs(fractions: dict[str, float], miff: np.ndarray) -> np.ndarray:
    density, elements = mixture_material(fractions)
    ticks = {name: round(elements[name] * 1_000_000_000_000) for name in ("H", "C", "N")}
    ticks["O"] = 1_000_000_000_000 - sum(ticks.values())
    with tempfile.TemporaryDirectory(prefix="keele_metal_xs_") as temporary:
        scratch = Path(temporary)
        ff = scratch / "trial_ff.dat"
        np.savetxt(ff, miff, fmt="%.12e")
        stem = scratch / "trial"
        case = dict(thickness_mm="50", gap_mm="110", beam_radius_mm="0.1",
                    focus_from_entry_mm="320", density_scale="1", case="0")
        macro = make_macro(case, 1, stem)
        macro = re.sub(r"/det/setCustomMatDensity .*", f"/det/setCustomMatDensity {density:.12f}", macro)
        for name, ticks_value in ticks.items():
            macro = re.sub(rf"/det/setCustomMat{name}massfract .*",
                           f"/det/setCustomMat{name}massfract {ticks_value / 1e12:.12f}", macro)
        macro = macro.replace("data/keele_breast_g50.dat", str(ff))
        path = stem.with_suffix(".mac")
        path.write_text(macro)
        environment = os.environ.copy()
        environment["SAXS_DUMP_XS"] = "1"
        environment["SAXS_IMAGE_ONLY"] = "1"
        environment["SAXS_IMAGE_CSV"] = "1"
        environment["SAXS_IMAGE_PIXELS"] = "1000"
        environment["SAXS_IMAGE_PITCH_UM"] = "100"
        run = subprocess.run([str(G4_BINARY), str(path), "1"], cwd=BUILD,
                             env=environment, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        if run.returncode:
            raise RuntimeError(run.stdout[-3500:])
        rows = []
        for line in run.stdout.splitlines():
            match = re.search(r"SAXS_XS CustomMat ([\d.]+) phot=([\deE+.-]+) "
                              r"compt=([\deE+.-]+) Rayl=([\deE+.-]+)", line)
            if match:
                rows.append([float(item) for item in match.groups()])
        if len(rows) != 6:
            raise RuntimeError("Geant4 did not report six CustomMat XS rows")
        return np.asarray(rows)


def main() -> None:
    cases = {
        "g50": {"water": .385893, "fat": .510305, "collagen": .103802},
        "water_rich": {"water": .55, "fat": .35, "collagen": .10},
        "fat_rich": {"water": .32, "fat": .58, "collagen": .10},
    }
    result = {}
    for name, fractions in cases.items():
        miff = trial_miff(fractions)
        g4 = g4_xs(fractions, miff)
        metal = trial_physics(miff, fractions)
        estimated = np.asarray(metal["sample_xs_mm_inv"]).T
        relative = estimated / g4[:, 1:] - 1
        result[name] = {
            "fractions": fractions,
            "max_relative_error_phot": float(np.max(abs(relative[:, 0]))),
            "max_relative_error_compt": float(np.max(abs(relative[:, 1]))),
            "max_relative_error_rayleigh": float(np.max(abs(relative[:, 2]))),
            "relative_error_at_22kev": relative[3].tolist(),
        }
    path = HERE / "results" / "multi_transport" / "trial_xs_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
