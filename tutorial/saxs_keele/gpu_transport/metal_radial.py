"""Exact GPU-side sparse radial reduction for the Metal photon transporter."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import spmatrix

from geometry import SlabDetectorGeometry

HERE = Path(__file__).resolve().parent
MAGIC = 0x52414431
MAX_LINKS = 3


@lru_cache(maxsize=1)
def _forward_model():
    """Load composition physics only for functions that run a trial."""
    if __package__:
        from .metal_forward import physics_reference_path, trial_physics
    else:
        sys.path.insert(0, str(HERE.parent))
        from metal_forward import physics_reference_path, trial_physics
    return physics_reference_path, trial_physics


@lru_cache(maxsize=1)
def metal_devices() -> tuple[dict, ...]:
    """Return physical Metal devices exposed to the transport executable."""
    payload = subprocess.check_output(
        [str(HERE / "multi_transport"), "--list-devices"], text=True,
    )
    devices = tuple(json.loads(payload))
    if not devices:
        raise RuntimeError("No Metal devices available")
    return devices


def write_radial_map(operator: spmatrix, direct_mask: np.ndarray,
                     path: Path) -> dict[str, int]:
    """Serialize a <=3-links/pixel CSR operator for the Metal reducer."""
    matrix = operator.tocsc()
    pixels_squared = matrix.shape[1]
    pixels = int(round(np.sqrt(pixels_squared)))
    if pixels * pixels != pixels_squared:
        raise ValueError("Radial operator must describe a square detector")
    direct = np.asarray(direct_mask, dtype=np.uint8)
    if direct.shape != (pixels, pixels):
        raise ValueError("Direct-beam mask shape differs from radial operator")
    per_pixel = np.diff(matrix.indptr)
    if int(per_pixel.max(initial=0)) > MAX_LINKS:
        raise ValueError("Metal sparse reducer supports at most three links per pixel")
    indices = np.full((pixels_squared, MAX_LINKS), -1, dtype="<i2")
    weights = np.zeros((pixels_squared, MAX_LINKS), dtype="<f4")
    for pixel in np.flatnonzero(per_pixel):
        start, stop = matrix.indptr[pixel:pixel + 2]
        count = stop - start
        indices[pixel, :count] = matrix.indices[start:stop]
        weights[pixel, :count] = matrix.data[start:stop]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<4I", MAGIC, pixels, matrix.shape[0], MAX_LINKS))
        stream.write(indices.tobytes(order="C"))
        stream.write(weights.tobytes(order="C"))
        stream.write(direct.ravel(order="C").tobytes())
    return {"pixels": pixels, "bins": matrix.shape[0],
            "nonzero_links": int(matrix.nnz), "bytes": path.stat().st_size}


def run_trial_metal_radial(*, miff: np.ndarray,
                           transport_fractions: dict[str, float],
                           manifest: dict, photons: int, seed: int,
                           radial_map: Path,
                           device_index: int | None = None,
                           ) -> tuple[np.ndarray, int, dict]:
    """Run full transport and return pyFAI-equivalent radial sums directly."""
    physics_reference_path, trial_physics = _forward_model()
    reference_path = physics_reference_path(float(manifest["energy_kev"]))
    geometry = SlabDetectorGeometry.from_manifest(manifest)
    physics = trial_physics(miff, transport_fractions, reference_path)
    physics["sample_lateral_mm"] = float(manifest.get("sample_lateral_mm", 120.0))
    with tempfile.TemporaryDirectory(prefix="keele_metal_radial_") as temporary:
        scratch = Path(temporary)
        ff = scratch / "trial_ff.dat"
        table = scratch / "trial_physics.json"
        radial_path = scratch / "radial.raw"
        np.savetxt(ff, miff, fmt="%.12e")
        table.write_text(json.dumps(physics, separators=(",", ":")))
        command = [
            str(HERE / "multi_transport"), str(photons),
            *geometry.metal_arguments(), "1", "1", "1", "1",
            str(seed), str(table), str(ff), "--radial", str(radial_map),
            str(radial_path),
        ]
        environment = os.environ.copy()
        if device_index is not None:
            environment["KEELE_METAL_DEVICE_INDEX"] = str(device_index)
        summary = json.loads(subprocess.check_output(
            command, text=True, env=environment,
        ))
        radial = np.fromfile(radial_path, dtype=np.float64)
    if radial.size != summary["radial_bins"]:
        raise ValueError("Metal radial output length mismatch")
    if summary["interaction_cap_count"]:
        raise RuntimeError(f"Metal interaction cap reached: {summary}")
    return radial, int(summary["direct_count"]), summary


def run_trial_metal_radial_many(
    *, miff: np.ndarray, transport_fractions: dict[str, float], manifest: dict,
    photons: int, seeds: Iterable[int], radial_map: Path,
    device_indices: tuple[int, ...] | None = None,
    workers_per_device: int | None = None,
) -> list[tuple[np.ndarray, int, dict]]:
    """Run independent seeds concurrently and preserve their input order.

    Seeds are assigned round-robin across physical Metal devices.  Two host
    workers per Apple device are the measured single-device optimum; callers
    can set one worker per device on a fully occupied discrete-GPU system.
    """
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        return []
    if device_indices is None:
        device_indices = tuple(int(device["index"]) for device in metal_devices())
    if not device_indices or any(index < 0 for index in device_indices):
        raise ValueError("device_indices must contain nonnegative indices")
    if workers_per_device is None:
        # Two concurrent processes improve occupancy on the tested M4 Pro.
        # With multiple physical devices, one process per device avoids
        # unvalidated oversubscription and maps four seeds to four GPUs.
        workers_per_device = 2 if len(device_indices) == 1 else 1
    if workers_per_device < 1:
        raise ValueError("workers_per_device must be positive")
    # Warm the composition-dependent Geant4/Penelope cache once before worker
    # threads enter trial_physics concurrently.
    physics_reference_path, trial_physics = _forward_model()
    trial_physics(
        miff, transport_fractions,
        physics_reference_path(float(manifest["energy_kev"])),
    )

    def one(item: tuple[int, int]) -> tuple[np.ndarray, int, dict]:
        position, seed = item
        device = device_indices[position % len(device_indices)]
        return run_trial_metal_radial(
            miff=miff, transport_fractions=transport_fractions,
            manifest=manifest, photons=photons, seed=seed,
            radial_map=radial_map, device_index=device,
        )

    workers = min(len(seeds), len(device_indices) * workers_per_device)
    if workers == 1:
        return [one(item) for item in enumerate(seeds)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, enumerate(seeds)))
