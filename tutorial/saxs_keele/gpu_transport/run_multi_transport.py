"""Run a reproducible 50 mm g50 Metal image with no retained ROOT/raw files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark import DEFAULT_FF, HERE

sys.path.insert(0, str(HERE.parent))
from compare_breast_50mm_xrd_preprocessing import integrate_with_xrd_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photons", type=int, default=200_000_000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "multi_transport")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="keele_metal_image_") as temporary:
        raw = Path(temporary) / "image.raw"
        command = [str(HERE / "multi_transport"), str(args.photons), "50", "110", "50",
                   "0.1", "0.1", "320", "1", "1", "1", "1", str(args.seed),
                   str(HERE / "results" / "multi_physics_g50.json"), str(DEFAULT_FF), str(raw)]
        started = time.perf_counter()
        summary = json.loads(subprocess.check_output(command, text=True))
        summary["wall_seconds_with_process_startup"] = time.perf_counter() - started
        image = np.fromfile(raw, dtype=np.uint32).reshape(1000, 1000)
    if summary["interaction_cap_count"]:
        raise RuntimeError(f"Untransported capped histories: {summary}")
    record = integrate_with_xrd_preprocessing(
        image, pixel_um=100.0, wavelength_m=1.2398419843320026e-9 / 22.162917,
        front_face_distance_m=0.160, sample_thickness_mm=50.0,
        q_range=(1.0, 30.0), npt=256,
    )
    q = np.asarray(record["q_range"], dtype=float)
    intensity = np.asarray(record["radial_profile_data"], dtype=float)
    stem = f"metal_{args.photons // 1_000_000}m"
    np.savez_compressed(args.output_dir / f"{stem}.npz", image=image, q_nm_inv=q,
                        radial_intensity=intensity)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].imshow(np.log1p(image), cmap="magma", origin="lower",
                   extent=(-50, 50, -50, 50))
    axes[0].set(xlim=(-25, 25), ylim=(-25, 25), xlabel="x (mm)", ylabel="y (mm)",
                title=f"Metal, {args.photons/1e6:g}M incident photons")
    axes[1].plot(q, intensity)
    axes[1].set(xlim=(1, 30), xlabel=r"$q$ (nm$^{-1}$)",
                ylabel="XRD-preprocessing intensity", title="Thickness-adjusted radial profile")
    fig.savefig(args.output_dir / f"{stem}.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"image": str(args.output_dir / f"{stem}.png"), **summary}, indent=2))


if __name__ == "__main__":
    main()
