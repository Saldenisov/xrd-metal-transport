"""Compare Geant4 and Metal detector channels with photon processes ablated.

Diagnostic only: process inactivation changes the physics. No ROOT or raw
detector image is retained. Both engines still score the same 100 mm square.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from validate_water_100mm_cpu_metal import (BUILD, CHANNELS, G4, METAL,
                                              WATER_FF, geant4_macro,
                                              read_g4_channels)

HERE = Path(__file__).resolve().parent
DEFAULT_PHYSICS = (HERE / "results" / "multi_transport" /
                   "water_22p022_shell_exact_xs.json")
MODES = {
    "full": (),
    "transport_only": ("phot", "compt", "Rayl"),
    "no_photo": ("phot",),
    "compton_only": ("phot", "Rayl"),
    "rayleigh_only": ("phot", "compt"),
}


def run_geant4(directory: Path, photons: int, threads: int,
               mode: str, run_index: int = 100) -> np.ndarray:
    stem = directory / f"g4_{mode}"
    macro = stem.with_suffix(".mac")
    commands = "".join(f"/process/inactivate {name}\n" for name in MODES[mode])
    macro.write_text(geant4_macro(stem, photons, run_index).replace(
        "/run/initialize\n", "/run/initialize\n" + commands))
    env = os.environ.copy()
    env.update(SAXS_IMAGE_ONLY="1", SAXS_IMAGE_CSV="1",
               SAXS_IMAGE_PIXELS="256", SAXS_IMAGE_PITCH_UM="390.625")
    env.pop("SAXS_DUMP_XS", None)
    run = subprocess.run([str(G4), str(macro), str(threads)], cwd=BUILD, env=env,
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if run.returncode or "command not found" in run.stdout.lower():
        raise RuntimeError(run.stdout[-6000:])
    if stem.with_suffix(".root").exists():
        raise AssertionError("Geant4 unexpectedly created ROOT output")
    return read_g4_channels(stem.with_name(f"{stem.name}_h1_channels.csv"))


def run_metal(directory: Path, photons: int, physics_path: Path,
              mode: str, seed: int) -> np.ndarray:
    physics = json.loads(physics_path.read_text())
    physics["upstream_air_mm"] = 0.1
    disabled = set(MODES[mode])
    for index, process in enumerate(("phot", "compt", "Rayl")):
        if process in disabled:
            physics["air_xs_mm_inv"][index] = [0.0] * len(physics["energy_kev"])
    table = directory / "physics.json"
    table.write_text(json.dumps(physics, separators=(",", ":")))
    scales = ["0" if process in disabled else "1"
              for process in ("phot", "compt", "Rayl")]
    command = [str(METAL), str(photons), "100", "100", "50", "0.390625",
               "0.05", "1000000000", "1", *scales, str(seed), str(table),
               str(WATER_FF)]
    run = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    if run.returncode:
        raise RuntimeError(run.stderr[-6000:])
    metadata = json.loads(run.stdout)
    if metadata["interaction_cap_count"]:
        raise AssertionError(f"Metal interaction cap reached: {metadata}")
    return np.asarray(metadata["classes"][:4], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--photons", type=int, default=20_000_000)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--seed", type=int, default=925_103)
    parser.add_argument("--geant4-run-index", type=int, default=100)
    args = parser.parse_args()
    if (args.photons < 1 or args.photons >= 2**32 or args.threads < 1
            or args.geant4_run_index < 0):
        parser.error("Invalid photons or threads")
    with tempfile.TemporaryDirectory(prefix="keele_transport_ablation_") as path:
        directory = Path(path)
        cpu = run_geant4(directory, args.photons, args.threads, args.mode,
                         args.geant4_run_index)
        gpu = run_metal(directory, args.photons, args.physics, args.mode, args.seed)
    comparison = {}
    for name, c, g in zip(CHANNELS, cpu, gpu):
        comparison[name] = {
            "geant4": int(c), "metal": int(g),
            "relative_difference": float(g / c - 1) if c else None,
            "two_sample_poisson_z": float((g - c) / np.sqrt(g + c)) if g + c else None,
        }
    print(json.dumps({"mode": args.mode, "photons_per_backend": args.photons,
                      "geant4_run_index": args.geant4_run_index,
                      "metal_seed": args.seed,
                      "channels": comparison, "roots_created": 0}))


if __name__ == "__main__":
    main()
