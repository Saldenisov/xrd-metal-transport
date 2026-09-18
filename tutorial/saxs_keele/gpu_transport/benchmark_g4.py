"""Validate direct and single-Rayleigh Metal channels against 100 Geant4 runs.

Geant4 ROOTs and macros are written to a temporary directory and removed
automatically after each case; only CSV, JSON and PNG summaries are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

if __package__:
    from .benchmark import DEFAULT_FF, HERE, MU_AIR
else:
    from benchmark import DEFAULT_FF, HERE, MU_AIR


BUILD = HERE.parent.parent.parent / "build" / "saxs_keele"
G4_BINARY = BUILD / "saxs"


def make_macro(case: dict[str, str], photons: int, output: Path) -> str:
    thickness = float(case["thickness_mm"])
    gap = float(case["gap_mm"])
    radius = float(case["beam_radius_mm"])
    focus = float(case["focus_from_entry_mm"])
    density = 0.9794738076 * float(case["density_scale"])
    entry = 100 - thickness / 2
    detector_distance = thickness / 2 + gap + 0.15
    return f"""/random/setSeeds {550071 + int(case['case']) * 37} {770053 + int(case['case']) * 41}
/det/setCustomMatDensity {density:.12g}
/det/setCustomMatHmassfract 0.1074785105
/det/setCustomMatCmassfract 0.4680700880
/det/setCustomMatNmassfract 0.0189646254
/det/setCustomMatOmassfract 0.4054867761
/det/SetCustomMatFF data/keele_breast_g50.dat
/det/setPhantomMaterial 30
/det/setPhantomBox true
/det/setPhantomDiameter {thickness:.12g} mm
/det/setPhantomHeight 120. mm
/det/setPhantomZ 100. mm
/det/setSlits false
/det/setDetectorSize 141.421356 mm
/det/setDetectorThickness 0.3 mm
/det/setDetectorSampleDistance {detector_distance:.12g} mm
/phys/SelectPhysicsList empenelopeMI
/run/setCut 0.01 mm
/run/verbose 0
/run/initialize
/run/setfilenamesave {output}
/sd/setSK true
/gps/particle gamma
/gps/ene/type Mono
/gps/ene/mono 22.162917 keV
/gps/pos/type Plane
/gps/pos/shape Circle
/gps/pos/radius {radius:.12g} mm
/gps/pos/centre 0. 0. {entry - 0.1:.12g} mm
/gps/ang/type focused
/gps/ang/focuspoint 0. 0. {entry + focus:.12g} mm
/run/beamOn {photons}
"""


def g4_counts(root: Path) -> tuple[int, int, np.ndarray]:
    direct = rayleigh = 0
    q_counts = np.zeros(256, dtype=np.int64)
    for a in uproot.iterate(
        f"{root}:part", ["e", "posx", "posy", "momz", "type", "trackID", "NRi", "NCi", "NDi"],
        library="np", step_size="100 MB",
    ):
        active = ((a["type"] == 0) & (a["trackID"] == 1) & (a["momz"] > 0)
                  & (abs(a["posx"]) < 50) & (abs(a["posy"]) < 50))
        pure = active & (a["NCi"] == 0) & (a["NDi"] == 0)
        one_r = pure & (a["NRi"] == 1)
        direct += int(np.count_nonzero(pure & (a["NRi"] == 0)))
        rayleigh += int(np.count_nonzero(one_r))
        theta = np.arccos(np.clip(a["momz"][one_r], -1, 1))
        q = 4 * np.pi * a["e"][one_r] / 1.239841984 * np.sin(theta / 2)
        q_counts += np.histogram(q, bins=256, range=(0, 70))[0]
    return direct, rayleigh, q_counts


def metal_q(case: dict[str, str], photons: int) -> np.ndarray:
    command = [str(HERE / "single_rayleigh"), "gpu", str(photons),
               case["thickness_mm"], case["gap_mm"], "50", "70",
               case["beam_radius_mm"], case["focus_from_entry_mm"],
               case["mu_total_mm_inv"], case["mu_rayleigh_mm_inv"],
               str(MU_AIR), "22.162917", str(800031 + int(case["case"]) * 17),
               str(DEFAULT_FF)]
    result = json.loads(subprocess.check_output(command, text=True))
    return np.asarray(result["q_bins"], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "results" / "benchmark.csv")
    parser.add_argument("--photons", type=int, default=100_000)
    parser.add_argument("--metal-photons", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    cases = list(csv.DictReader(args.input.open()))
    source_photons = json.loads(args.input.with_name("summary.json").read_text())["photons_per_case"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    observed_q = np.zeros(256, dtype=float)
    expected_q = np.zeros(256, dtype=float)
    with tempfile.TemporaryDirectory(prefix="keele_g4_metal_") as temp:
        scratch = Path(temp)
        for index, case in enumerate(cases):
            stem = scratch / f"case_{index:03d}"
            macro = stem.with_suffix(".mac")
            macro.write_text(make_macro(case, args.photons, stem))
            environment = os.environ.copy()
            environment.pop("SAXS_IMAGE_ONLY", None)
            environment.pop("SAXS_DUMP_XS", None)
            process = subprocess.run(
                [str(G4_BINARY), str(macro), str(args.threads)],
                cwd=BUILD, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if process.returncode:
                raise RuntimeError(f"Geant4 case {index} failed:\n{process.stdout[-2000:]}")
            root = stem.with_suffix(".root")
            direct, rayleigh, q = g4_counts(root)
            root.unlink()
            macro.unlink()
            pred_direct = args.photons * int(case["cpu_direct"]) / source_photons
            pred_rayleigh = args.photons * int(case["cpu_single_rayleigh"]) / source_photons
            expected_q += metal_q(case, args.metal_photons) * args.photons / args.metal_photons
            observed_q += q
            records.append({"case": index, "g4_direct": direct, "metal_direct_expected": pred_direct,
                            "g4_rayleigh": rayleigh, "metal_rayleigh_expected": pred_rayleigh,
                            "direct_z": (direct - pred_direct) / np.sqrt(max(1, pred_direct)),
                            "rayleigh_z": (rayleigh - pred_rayleigh) / np.sqrt(max(1, pred_rayleigh))})
            if (index + 1) % 10 == 0:
                print(f"Geant4 {index + 1}/{len(cases)}", flush=True)

    with (args.output_dir / "geant4_100case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    centres = (np.arange(256) + 0.5) * 70 / 256
    # Geant4 has N histories while the Metal reference has M. After scaling
    # Metal counts by N/M, its variance is (N/M) * expected_q, not expected_q.
    reference_ratio = args.photons / args.metal_photons
    sigma = np.sqrt(np.maximum(1, observed_q + reference_ratio * expected_q))
    visible = (centres < 60) & (observed_q + expected_q > 20)
    q_chi2 = float(np.mean(((observed_q[visible] - expected_q[visible]) / sigma[visible]) ** 2))
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].step(centres, observed_q, where="mid", label="Geant4, 100 geometries", color="#111827")
    axes[0].step(centres, expected_q, where="mid", label="Metal, single Rayleigh", color="#ef4444")
    axes[0].set(ylabel="Pooled photons / q bin", title="Independent Geant4 versus Metal, pooled geometry test")
    axes[0].legend()
    axes[1].plot(centres, (observed_q - expected_q) / sigma, linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlim=(0, 60), ylim=(-4, 4), xlabel=r"$q$ (nm$^{-1}$)", ylabel="Residual / σ")
    fig.savefig(args.output_dir / "geant4_100case_q.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(args.output_dir / "geant4_100case_q.npz", q_nm_inv=centres,
                        geant4_counts=observed_q, metal_expected_counts=expected_q)
    observed_direct = sum(x["g4_direct"] for x in records)
    expected_direct = sum(x["metal_direct_expected"] for x in records)
    observed_rayleigh = sum(x["g4_rayleigh"] for x in records)
    expected_rayleigh = sum(x["metal_rayleigh_expected"] for x in records)
    summary = {
        "cases": len(cases), "g4_photons_per_case": args.photons,
        "g4_total_photons": len(cases) * args.photons,
        "mean_direct_z": float(np.mean([x["direct_z"] for x in records])),
        "rms_direct_z": float(np.sqrt(np.mean([x["direct_z"] ** 2 for x in records]))),
        "mean_rayleigh_z": float(np.mean([x["rayleigh_z"] for x in records])),
        "rms_rayleigh_z": float(np.sqrt(np.mean([x["rayleigh_z"] ** 2 for x in records]))),
        "pooled_q_reduced_chi2": q_chi2,
        "q_compared_bins": int(np.count_nonzero(visible)),
        "direct_relative_bias_percent": 100 * (observed_direct / expected_direct - 1),
        "rayleigh_relative_bias_percent": 100 * (observed_rayleigh / expected_rayleigh - 1),
        "g4_roots_retained": 0,
        "physics_scope": "direct and exactly-one-Rayleigh only",
    }
    (args.output_dir / "geant4_100case_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
