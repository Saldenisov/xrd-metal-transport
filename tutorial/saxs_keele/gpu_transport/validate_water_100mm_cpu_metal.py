"""Independent Geant4/Metal water-slab parity test without ROOT files.

Both backends transport a parallel 100-um-diameter circular beam through a
100-mm homogeneous water slab and 100 mm of downstream air. The detector is an
ideal 1000x1000 entrance counter with 100-um pixels. XRD-preprocessing gives
the same 200-point, thickness-adjusted q profile for both backends. Its pyFAI
pixel-splitting CSR operator also supplies the full inter-bin noise covariance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import chi2
from xrd_preprocessing.azimuthal import _integrator_from_dataframe

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
BUILD = HERE.parents[2] / "build" / "saxs_keele"
G4 = BUILD / "saxs"
METAL = HERE / "multi_transport"
WATER_FF = SOURCE / "data" / "keele_water.dat"
WATER_PHYSICS = HERE / "results" / "multi_transport" / "water_22p022_physics.json"
from profile_integration import integrate_image

ENERGY_KEV = 22.0220601
THICKNESS_MM = 100.0
GAP_MM = 100.0
PIXELS = 1000
PITCH_UM = 100.0
NPT = 200
MANIFEST = {
    "detector_pixels": PIXELS,
    "pixel_pitch_um": PITCH_UM,
    "sample_thickness_mm": THICKNESS_MM,
    "sample_lateral_mm": 120.0,
    "front_face_to_detector_mm": THICKNESS_MM + GAP_MM,
    "mean_scattering_plane_to_detector_mm": GAP_MM + THICKNESS_MM / 2,
    "energy_kev": ENERGY_KEV,
    "q_range_nm_inv": [1.0, 30.0],
    "radial_points": NPT,
}
CHANNELS = ("direct", "single_rayleigh", "multiple_rayleigh", "compton")


def geant4_macro(output: Path, photons: int, run_index: int) -> str:
    first_seed = 600_031 + 1021 * run_index
    second_seed = 700_033 + 1031 * run_index
    return f"""/random/setSeeds {first_seed} {second_seed}
/det/setCustomMatDensity 1.000000000000
/det/setCustomMatHmassfract 0.111900000000
/det/setCustomMatCmassfract 0.000000000000
/det/setCustomMatNmassfract 0.000000000000
/det/setCustomMatOmassfract 0.888100000000
/det/SetCustomMatFF data/keele_water.dat
/det/setPhantomMaterial 30
/det/setPhantomBox true
/det/setPhantomDiameter 100. mm
/det/setPhantomHeight 120. mm
/det/setPhantomZ 100. mm
/det/setSlits false
/det/setDetectorSize 141.421356 mm
/det/setDetectorThickness 0.3 mm
/det/setDetectorSampleDistance 150.15 mm
/phys/SelectPhysicsList empenelopeMI
/run/setCut 0.01 mm
/run/verbose 0
/run/initialize
/run/setfilenamesave {output}
/sd/setSK true
/gps/particle gamma
/gps/ene/type Mono
/gps/ene/mono {ENERGY_KEV:.7f} keV
/gps/pos/type Plane
/gps/pos/shape Circle
/gps/pos/radius 0.05 mm
/gps/pos/centre 0. 0. 49.9 mm
/gps/direction 0 0 1
/run/printProgress 50000000
/run/beamOn {photons}
"""


def read_g4_image(path: Path) -> np.ndarray:
    with path.open(encoding="ascii") as stream:
        header = [next(stream).strip() for _ in range(10)]
    if header[0] != "#class tools::histo::h2d":
        raise ValueError(f"Unexpected Geant4 image header: {path}")
    values = np.loadtxt(path, delimiter=",", skiprows=10, usecols=(0,))
    if values.size != (PIXELS + 2) ** 2:
        raise ValueError(f"Unexpected Geant4 image length: {values.size}")
    image = values.reshape(PIXELS + 2, PIXELS + 2)[1:-1, 1:-1]
    if not np.allclose(image, np.rint(image)):
        raise ValueError("Geant4 image has noninteger counts")
    return np.rint(image).astype(np.int64)


def read_g4_channels(path: Path) -> np.ndarray:
    with path.open(encoding="ascii") as stream:
        rows = [line.strip() for line in stream if line.strip() and not line.startswith("#")]
    values = []
    for row in rows:
        try:
            values.append(float(row.split(",")[0]))
        except ValueError:
            continue  # Geant4 CSV column-name row.
    values = np.asarray(values)
    if values.size != len(CHANNELS) + 2:
        raise ValueError(f"Unexpected Geant4 channel histogram: {path}: {values}")
    result = values[1:-1]
    if not np.allclose(result, np.rint(result)):
        raise ValueError("Geant4 channel counts are not integers")
    return np.rint(result).astype(np.int64)


def run_geant4(scratch: Path, photons: int, index: int, threads: int) -> tuple[np.ndarray, np.ndarray, float]:
    stem = scratch / f"cpu_{index:02d}"
    macro = stem.with_suffix(".mac")
    macro.write_text(geant4_macro(stem, photons, index))
    env = os.environ.copy()
    env["SAXS_IMAGE_ONLY"] = "1"
    env["SAXS_IMAGE_CSV"] = "1"
    env["SAXS_IMAGE_PIXELS"] = str(PIXELS)
    env["SAXS_IMAGE_PITCH_UM"] = str(PITCH_UM)
    env.pop("SAXS_DUMP_XS", None)
    started = time.perf_counter()
    run = subprocess.run([str(G4), str(macro), str(threads)], cwd=BUILD, env=env,
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(run.stdout[-6000:])
    if stem.with_suffix(".root").exists():
        raise AssertionError("Geant4 unexpectedly created ROOT output")
    image = read_g4_image(stem.with_name(f"{stem.name}_h2_image.csv"))
    channels = read_g4_channels(stem.with_name(f"{stem.name}_h1_channels.csv"))
    if int(channels.sum()) != int(image.sum()):
        raise AssertionError(f"Geant4 class and image sums differ: {channels.sum()} != {image.sum()}")
    return image, channels, elapsed


def metal_seed(photons: int, index: int) -> int:
    """Distinct 32-bit Philox keys for independent benchmark repetitions."""
    return (800_053 + index * photons * 747_796_405) % (2**32)


def run_metal(scratch: Path, photons: int, index: int,
              physics_path: Path = WATER_PHYSICS) -> tuple[np.ndarray, np.ndarray, dict, float]:
    physics = json.loads(physics_path.read_text())
    physics["upstream_air_mm"] = 0.1
    table = scratch / "water_physics.json"
    table.write_text(json.dumps(physics, separators=(",", ":")))
    raw = scratch / f"gpu_{index:02d}.raw"
    # Each Philox counter contains the photon history index and draw-block
    # number; each repetition receives a distinct key.
    seed = metal_seed(photons, index)
    command = [str(METAL), str(photons), "100", "100", "50", "0.1", "0.05",
               "1000000000", "1", "1", "1", "1", str(seed), str(table),
               str(WATER_FF), str(raw)]
    started = time.perf_counter()
    run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(run.stderr[-6000:])
    metadata = json.loads(run.stdout)
    metadata["history_seed"] = seed
    if metadata["interaction_cap_count"]:
        raise AssertionError("Metal interaction cap reached")
    image = np.fromfile(raw, dtype=np.uint32).reshape(PIXELS, PIXELS).astype(np.int64)
    channels = np.asarray(metadata["classes"][:4], dtype=np.int64)
    if int(channels.sum()) != int(image.sum()):
        raise AssertionError("Metal class and image sums differ")
    return image, channels, metadata, elapsed


def pyfai_operator() -> tuple[np.ndarray, csr_matrix]:
    wavelength_nm = 1.2398419843320026 / ENERGY_KEV
    ai = _integrator_from_dataframe(PITCH_UM * 1e-6, PIXELS / 2, PIXELS / 2,
                                    wavelength_nm * 10, GAP_MM + THICKNESS_MM / 2)
    result = ai.integrate1d(np.zeros((PIXELS, PIXELS), dtype=np.float32), NPT,
                            unit="q_nm^-1", radial_range=(1.0, 30.0),
                            error_model="poisson", correctSolidAngle=False)
    engine = ai.engines[result.method].engine
    weights = csr_matrix((engine.data, engine.indices, engine.indptr),
                         shape=(NPT, PIXELS * PIXELS))
    return np.asarray(result.radial), weights


def profile_parity(cpu_image: np.ndarray, gpu_image: np.ndarray,
                   q: np.ndarray, weights: csr_matrix) -> dict:
    cpu = np.asarray(weights @ cpu_image.ravel(), dtype=float)
    gpu = np.asarray(weights @ gpu_image.ravel(), dtype=float)
    difference = gpu - cpu
    pooled_pixel_counts = cpu_image.ravel() + gpu_image.ravel()
    covariance = (weights.multiply(pooled_pixel_counts) @ weights.T).toarray()
    variance = np.diag(covariance)
    active = variance > 0
    if active.sum() < 2:
        raise ValueError("Too few observed q bins for the parity test")
    covariance_active = covariance[np.ix_(active, active)]
    difference_active = difference[active]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_active)
    threshold = eigenvalues[-1] * 1e-10
    resolved = eigenvalues > threshold
    projections = eigenvectors[:, resolved].T @ difference_active
    chisquare = float(np.sum(projections**2 / eigenvalues[resolved]))
    dof = int(np.count_nonzero(resolved))
    z = np.divide(difference, np.sqrt(variance), out=np.zeros_like(difference),
                  where=variance > 0)
    return {
        "bins": int(active.sum()), "covariance_rank": dof,
        "chi2": chisquare, "reduced_chi2": chisquare / dof,
        "chi2_p_value": float(chi2.sf(chisquare, dof)),
        "max_abs_bin_z": float(np.max(np.abs(z[active]))),
        "mean_bin_z": float(np.mean(z[active])),
        "integrated_relative_difference": float(gpu.sum() / cpu.sum() - 1),
        "q_min_nm_inv": float(q[0]), "q_max_nm_inv": float(q[-1]),
        "cpu_profile": cpu, "gpu_profile": gpu, "bin_z": z,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photons", type=int, default=200_000_000)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--seed-run-offset", type=int, default=0,
                        help="Offset CPU and GPU run indices for independent histories")
    parser.add_argument("--metal-physics", type=Path, default=WATER_PHYSICS)
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "results" / "water_100mm_cpu_metal_10x200m")
    args = parser.parse_args()
    if min(args.photons, args.repetitions, args.threads) < 1:
        parser.error("Photons, repetitions and threads must be positive")
    if args.seed_run_offset < 0:
        parser.error("--seed-run-offset must be nonnegative")
    if args.photons >= 2**32:
        parser.error("The current Philox counter stores a run's history ID in 32 bits")
    if (args.output_dir / "summary.json").exists():
        parser.error(f"Output already exists: {args.output_dir / 'summary.json'}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metal_physics = json.loads(args.metal_physics.read_text())
    if metal_physics.get("sample_lateral_mm", 120.0) != MANIFEST["sample_lateral_mm"]:
        parser.error("Metal sample lateral dimension differs from Geant4 box")
    q_operator, weights = pyfai_operator()
    pooled_cpu = np.zeros((PIXELS, PIXELS), dtype=np.int64)
    pooled_gpu = np.zeros_like(pooled_cpu)
    cpu_channels = np.zeros(4, dtype=np.int64)
    gpu_channels = np.zeros(4, dtype=np.int64)
    records = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="keele_water_parity_") as temporary:
        scratch = Path(temporary)
        for index in range(args.repetitions):
            run_index = index + args.seed_run_offset
            cpu_image, cpu_count, cpu_seconds = run_geant4(
                scratch, args.photons, run_index, args.threads)
            gpu_image, gpu_count, gpu_info, gpu_seconds = run_metal(
                scratch, args.photons, run_index, args.metal_physics)
            q_cpu, cpu_profile, cpu_distance = integrate_image(cpu_image, MANIFEST)
            q_gpu, gpu_profile, gpu_distance = integrate_image(gpu_image, MANIFEST)
            if not (np.array_equal(q_cpu, q_gpu) and np.array_equal(q_cpu, q_operator)
                    and cpu_distance == 150.0 and gpu_distance == 150.0):
                raise AssertionError("CPU/Metal XRD-preprocessing geometry differs")
            if not np.allclose(weights @ cpu_image.ravel(), cpu_profile, rtol=1e-9, atol=1e-7):
                raise AssertionError("Geant4 pyFAI CSR and XRD-preprocessing differ")
            if not np.allclose(weights @ gpu_image.ravel(), gpu_profile, rtol=1e-9, atol=1e-7):
                raise AssertionError("Metal pyFAI CSR and XRD-preprocessing differ")
            pooled_cpu += cpu_image
            pooled_gpu += gpu_image
            cpu_channels += cpu_count
            gpu_channels += gpu_count
            records.append({"run": run_index, "incident_photons_per_backend": args.photons,
                            "geant4_seconds": cpu_seconds, "metal_seconds": gpu_seconds,
                            "geant4_channels": cpu_count.tolist(),
                            "metal_channels": gpu_count.tolist(),
                            "metal_history_seed": gpu_info["history_seed"],
                            "metal_air_interaction_hits": gpu_info["air_interaction_hits"],
                            "profile_integrated_cpu": float(cpu_profile.sum()),
                            "profile_integrated_gpu": float(gpu_profile.sum())})
            print(json.dumps(records[-1]), flush=True)
            # The scratch files are not needed after each 1D integration.
            for path in scratch.glob(f"cpu_{run_index:02d}*"):
                path.unlink()
            (scratch / f"gpu_{run_index:02d}.raw").unlink()
    parity = profile_parity(pooled_cpu, pooled_gpu, q_operator, weights)
    pooled = {}
    for channel, cpu_count, gpu_count in zip(CHANNELS, cpu_channels, gpu_channels):
        pooled[channel] = {
            "geant4": int(cpu_count), "metal": int(gpu_count),
            "relative_difference": float(gpu_count / cpu_count - 1),
            "poisson_z": float((gpu_count - cpu_count) / np.sqrt(gpu_count + cpu_count)),
        }
    total = args.photons * args.repetitions
    summary = {
        "geometry": MANIFEST, "beam_diameter_um": 100.0,
        "beam_direction": "parallel", "upstream_air_mm": 0.1,
        "downstream_air_mm": GAP_MM, "incident_photons_per_backend": total,
        "repetitions": args.repetitions, "seed_run_offset": args.seed_run_offset,
        "individual_runs": records,
        "channels": pooled,
        "transmission_geant4": float(cpu_channels[0] / total),
        "transmission_metal": float(gpu_channels[0] / total),
        "profile_parity": {key: value for key, value in parity.items()
                           if not isinstance(value, np.ndarray)},
        "geant4_source_sha256_16": digest(SOURCE / "src" / "SAXSEventAction.cc"),
        "metal_source_sha256_16": digest(HERE / "multi_transport.swift"),
        "metal_physics_sha256_16": digest(args.metal_physics),
        "metal_physics_path": str(args.metal_physics),
        "roots_created": 0, "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["ideal detector", "homogeneous water", "no fluorescence",
                        "six-node cross-section interpolation after Compton"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Retain the small 200x200 CPU covariance so later GPU-only revisions can
    # be compared exactly without keeping any detector image or ROOT file.
    cpu_covariance = (weights.multiply(pooled_cpu.ravel()) @ weights.T).toarray()
    np.savez_compressed(args.output_dir / "profiles.npz", q_nm_inv=q_operator,
                        geant4=parity["cpu_profile"], metal=parity["gpu_profile"],
                        geant4_covariance=cpu_covariance, bin_z=parity["bin_z"])
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]},
                             constrained_layout=True)
    axes[0].plot(q_operator, parity["cpu_profile"] / total, label="Geant4 CPU", lw=1.5)
    axes[0].plot(q_operator, parity["gpu_profile"] / total, label="Metal GPU", lw=1.0)
    axes[0].set(ylabel="XRD-preprocessing sum_signal / incident photon", yscale="log")
    axes[0].legend(frameon=False)
    axes[1].plot(q_operator, parity["bin_z"], lw=0.9)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhline(3, color="gray", lw=0.7, ls="--")
    axes[1].axhline(-3, color="gray", lw=0.7, ls="--")
    axes[1].set(xlabel=r"$q$ (nm$^{-1}$)", ylabel="Metal − Geant4 / σ", xlim=(1, 30))
    figure = args.output_dir / "profile_comparison.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"),
                      "figure": str(figure),
                      "reduced_chi2": parity["reduced_chi2"],
                      "chi2_p_value": parity["chi2_p_value"],
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
