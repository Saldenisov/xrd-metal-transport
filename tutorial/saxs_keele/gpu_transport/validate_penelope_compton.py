"""Compare Metal Penelope shell/Doppler Compton final states with Geant4.

The reference histograms come from temporary Geant4 event output produced by
prepare_multi_physics.py; no ROOT or raw photon histories are retained.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
METAL = HERE / "multi_transport"


def compare_hist(cpu: np.ndarray, gpu: np.ndarray) -> dict:
    active = cpu + gpu > 0
    statistic = float(np.sum((cpu[active] - gpu[active]) ** 2 /
                             (cpu[active] + gpu[active])))
    dof = max(int(active.sum()) - 1, 1)
    return {"chi2": statistic, "dof": dof, "reduced_chi2": statistic / dof,
            "p_value": float(chi2.sf(statistic, dof))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=97_031)
    args = parser.parse_args()
    physics = json.loads(args.physics.read_text())
    reference = physics["compton_probe_reference"]
    n = int(reference["count"])
    with tempfile.TemporaryDirectory(prefix="keele_compton_probe_") as temporary:
        output = Path(temporary) / "metal.raw"
        command = [str(METAL), "--probe-compton", str(n), str(args.seed),
                   str(physics["source_energy_kev"]), str(args.physics),
                   "sample", str(output)]
        run = subprocess.run(command, capture_output=True, text=True, check=True)
        result = np.fromfile(output, dtype="<f4").reshape(-1, 2).astype(float)
    if result.shape != (n, 2) or not np.isfinite(result).all() or np.any(result[:, 0] < 0):
        raise AssertionError("Metal Compton final state has failed or nonfinite events")
    angle_edges = np.linspace(0, np.pi, 65)
    energy_edges = np.linspace(0, physics["source_energy_kev"] * 1.001, 65)
    metal_theta_hist = np.histogram(result[:, 0], bins=angle_edges)[0]
    metal_energy_hist = np.histogram(result[:, 1], bins=energy_edges)[0]
    g4_theta_hist = np.asarray(reference["theta_hist_64"], dtype=np.int64)
    g4_energy_hist = np.asarray(reference["e_after_hist_64"], dtype=np.int64)
    comparisons = {}
    for name, column, cpu_mean, cpu_sd, cpu_hist, gpu_hist in (
        ("theta_rad", result[:, 0], reference["theta_mean_rad"],
         reference["theta_sd_rad"], g4_theta_hist, metal_theta_hist),
        ("e_after_kev", result[:, 1], reference["e_after_mean_kev"],
         reference["e_after_sd_kev"], g4_energy_hist, metal_energy_hist),
    ):
        gpu_mean = float(column.mean())
        gpu_sd = float(column.std())
        z = (gpu_mean - cpu_mean) / np.sqrt((cpu_sd**2 + gpu_sd**2) / n)
        comparisons[name] = {"geant4_mean": cpu_mean, "metal_mean": gpu_mean,
                             "mean_difference_z": float(z),
                             "histogram": compare_hist(cpu_hist, gpu_hist)}
    if "theta_energy_hist_32x32" in reference:
        cpu_joint = np.asarray(reference["theta_energy_hist_32x32"], dtype=np.int64)
        gpu_joint = np.histogram2d(
            result[:, 0], result[:, 1],
            bins=(np.linspace(0, np.pi, 33),
                  np.linspace(0, physics["source_energy_kev"] * 1.001, 33)))[0]
        comparisons["theta_energy_joint"] = compare_hist(cpu_joint.ravel(), gpu_joint.ravel())
    summary = {"incident_energy_kev": physics["source_energy_kev"],
               "independent_compton_events_per_backend": n,
               "metal_kernel": json.loads(run.stdout),
               "comparisons": comparisons,
               "roots_retained": 0}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, edges, cpu, gpu, title in (
        (axes[0], angle_edges, g4_theta_hist, metal_theta_hist, "Compton angle (rad)"),
        (axes[1], energy_edges, g4_energy_hist, metal_energy_hist, "Outgoing photon energy (keV)"),
    ):
        centers = (edges[1:] + edges[:-1]) / 2
        ax.step(centers, cpu, where="mid", label="Geant4 Penelope")
        ax.step(centers, gpu, where="mid", label="Metal Penelope port", alpha=0.8)
        ax.set_xlabel(title)
        ax.set_ylabel("Counts")
        ax.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
