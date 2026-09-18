"""Replace a Metal water XS node or audit its interpolation against Geant4.

Uses one Geant4 history only to initialize the physics list and query
G4EmCalculator and the tracking lambda table. Image-only CSV output is
temporary. No ROOT file is made.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

from validate_water_100mm_cpu_metal import BUILD, ENERGY_KEV, G4, WATER_PHYSICS, geant4_macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-energy-kev", type=float)
    parser.add_argument("--physics", type=Path, default=WATER_PHYSICS)
    args = parser.parse_args()
    if (args.output is None) == (args.audit_energy_kev is None):
        parser.error("Specify exactly one of --output and --audit-energy-kev")
    if args.audit_energy_kev is not None and args.audit_energy_kev <= 0:
        parser.error("--audit-energy-kev must be positive")
    physics = json.loads(args.physics.read_text())
    grid = physics["energy_kev"]
    replace = min(range(len(grid)), key=lambda i: abs(grid[i] - ENERGY_KEV))
    if args.audit_energy_kev is None and not (grid[replace - 1] < ENERGY_KEV < grid[replace + 1]):
        raise ValueError("Exact energy cannot replace the selected interior grid node")
    with tempfile.TemporaryDirectory(prefix="keele_exact_xs_") as directory:
        stem = Path(directory) / "water_exact"
        macro = stem.with_suffix(".mac")
        macro.write_text(geant4_macro(stem, 1, 0))
        env = os.environ.copy()
        env["SAXS_IMAGE_ONLY"] = "1"
        env["SAXS_IMAGE_CSV"] = "1"
        env["SAXS_IMAGE_PIXELS"] = "16"
        env["SAXS_IMAGE_PITCH_UM"] = "6250"
        env["SAXS_DUMP_XS"] = "1"
        query_energy = (args.audit_energy_kev if args.audit_energy_kev is not None
                        else ENERGY_KEV)
        env["SAXS_DUMP_XS_EXTRA_KEV"] = f"{query_energy:.9f}"
        result = subprocess.run([str(G4), str(macro), "1"], cwd=BUILD, env=env,
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=True)
        if list(Path(directory).glob("*.root")):
            raise AssertionError("Unexpected ROOT output")
    pattern = re.compile(r"SAXS_XS (CustomMat|Air) ([\d.]+) phot=([\deE+.-]+) "
                         r"compt=([\deE+.-]+) Rayl=([\deE+.-]+)")
    exact: dict[str, tuple[float, float, float]] = {}
    tracking: dict[str, tuple[float, float, float]] = {}
    for match in pattern.finditer(result.stdout):
        if abs(float(match.group(2)) - query_energy) < 1e-6:
            exact[match.group(1)] = tuple(map(float, match.groups()[2:]))
    for match in re.finditer(pattern.pattern.replace("SAXS_XS", "SAXS_TRACK_XS"),
                             result.stdout):
        if abs(float(match.group(2)) - query_energy) < 1e-6:
            tracking[match.group(1)] = tuple(map(float, match.groups()[2:]))
    if set(exact) != {"CustomMat", "Air"}:
        raise RuntimeError("Missing exact Geant4 cross sections")
    if set(tracking) != {"CustomMat", "Air"}:
        raise RuntimeError("Missing Geant4 tracking cross sections")
    if args.audit_energy_kev is not None:
        if not (grid[0] <= query_energy <= grid[-1]):
            raise ValueError("Audit energy lies outside the Metal cross-section grid")
        index = min(range(len(grid) - 1), key=lambda i: abs(grid[i] - query_energy)
                    if grid[i] <= query_energy <= grid[i + 1] else math.inf)
        fraction = (query_energy - grid[index]) / (grid[index + 1] - grid[index])
        results = {}
        for material, key in (("CustomMat", "sample_xs_mm_inv"),
                              ("Air", "air_xs_mm_inv")):
            results[material] = {}
            for process, name in enumerate(("phot", "compt", "Rayl")):
                nodes = physics[key][process]
                interpolated = math.exp((1 - fraction) * math.log(nodes[index]) +
                                        fraction * math.log(nodes[index + 1]))
                results[material][name] = {
                    "geant4_mm_inv": exact[material][process],
                    "geant4_tracking_mm_inv": tracking[material][process],
                    "metal_interpolated_mm_inv": interpolated,
                    "relative_error": interpolated / exact[material][process] - 1,
                    "relative_error_to_tracking":
                        interpolated / tracking[material][process] - 1,
                }
        print(json.dumps({"energy_kev": query_energy, "cross_sections": results,
                          "roots_created": 0}))
        return
    physics["energy_kev"][replace] = ENERGY_KEV
    for material, key in (("CustomMat", "sample_xs_mm_inv"),
                          ("Air", "air_xs_mm_inv")):
        for process in range(3):
            physics[key][process][replace] = exact[material][process]
    physics["source"] += "; exact incident-energy G4EmCalculator XS"
    physics["exact_energy_xs_kev"] = ENERGY_KEV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(physics, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "energy_kev": ENERGY_KEV,
                      "sample_xs_mm_inv": exact["CustomMat"],
                      "air_xs_mm_inv": exact["Air"], "roots_created": 0}))


if __name__ == "__main__":
    main()
