"""High-statistics Metal Penelope port check against retained 2B Geant4 water CPU run.

The Geant4 photon histories were already executed; this script does not
pretend to reconstruct their discarded detector-image covariance. A separate
fresh 200M+200M test provides the exact CSR-covariance profile comparison.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validate_water_100mm_cpu_metal import CHANNELS, pyfai_operator, run_metal

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "results" / "water_100mm_cpu_metal_10x200m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--photons", type=int, default=200_000_000)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed-run-offset", type=int, default=0,
                        help="Offset Metal run indices to obtain independent Philox keys")
    args = parser.parse_args()
    if args.seed_run_offset < 0:
        parser.error("--seed-run-offset must be nonnegative")
    reference = json.loads((REFERENCE / "summary.json").read_text())
    if args.photons * args.repetitions != reference["incident_photons_per_backend"]:
        parser.error("Metal photon count must match retained Geant4 reference")
    if (args.output_dir / "summary.json").exists():
        parser.error("Output summary already exists")
    q, weights = pyfai_operator()
    pooled_image = np.zeros((1000, 1000), dtype=np.int64)
    channels = np.zeros(4, dtype=np.int64)
    seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="keele_retained_cpu_port_") as temporary:
        scratch = Path(temporary)
        for index in range(args.repetitions):
            run_index = index + args.seed_run_offset
            image, detected, metadata, elapsed = run_metal(scratch, args.photons,
                                                            run_index,
                                                            args.physics)
            pooled_image += image
            channels += detected
            seconds += elapsed
            (scratch / f"gpu_{run_index:02d}.raw").unlink()
            print(json.dumps({"run": run_index,
                              "metal_channels": detected.tolist(),
                              "seconds": elapsed}), flush=True)
    comparison = {}
    for index, channel in enumerate(CHANNELS):
        cpu = reference["channels"][channel]["geant4"]
        gpu = int(channels[index])
        comparison[channel] = {"geant4": cpu, "metal": gpu,
                               "relative_difference": gpu / cpu - 1,
                               "poisson_z": (gpu - cpu) / np.sqrt(gpu + cpu)}
    cpu_profile = np.load(REFERENCE / "profiles.npz")["geant4"]
    gpu_profile = np.asarray(weights @ pooled_image.ravel())
    summary = {"incident_photons_per_backend": args.photons * args.repetitions,
               "repetitions": args.repetitions, "seed_run_offset": args.seed_run_offset,
               "channels": comparison,
               "profile_integrated_geant4": float(cpu_profile.sum()),
               "profile_integrated_metal": float(gpu_profile.sum()),
               "profile_integrated_relative_difference": float(gpu_profile.sum() /
                                                                cpu_profile.sum() - 1),
               "profile_caveat": "Geant4 pixel image discarded; exact inter-bin covariance "
                                 "unavailable, so no formal profile p-value is reported",
               "metal_elapsed_seconds": seconds, "roots_retained": 0}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "profiles.npz", q_nm_inv=q,
                        geant4=cpu_profile, metal=gpu_profile)
    figure, ax = plt.subplots(figsize=(8, 4))
    ax.plot(q, cpu_profile, label="Geant4 CPU, retained 2B")
    ax.plot(q, gpu_profile, label="Metal Penelope port, independent 2B", alpha=0.8)
    ax.set(xlabel=r"$q$ (nm$^{-1}$)", ylabel="pyFAI sum_signal")
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "profile_comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
