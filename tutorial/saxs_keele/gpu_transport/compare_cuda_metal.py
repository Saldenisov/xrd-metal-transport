#!/usr/bin/env python3
"""Run identical histories through native Metal and CUDA translated by CuMetal."""

from __future__ import annotations

import argparse
from array import array
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_PHYSICS = HERE / "results/multi_transport/g50_22p163_shell_joint.json"
DEFAULT_FORM_FACTOR = HERE.parent / "data/keele_breast_g50.dat"


def command_json(command: list[str]) -> dict:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"Command returned no JSON summary: {' '.join(command)}")
    return json.loads(lines[-1])


def load_u32(path: Path) -> array:
    result = array("I")
    result.frombytes(path.read_bytes())
    if result.itemsize != 4:
        raise RuntimeError("This validation requires 32-bit unsigned integers")
    return result


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photons", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20_260_917)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--form-factor", type=Path, default=DEFAULT_FORM_FACTOR)
    parser.add_argument("--thickness-mm", type=float, default=50.0)
    parser.add_argument("--gap-mm", type=float, default=110.0)
    parser.add_argument("--detector-half-mm", type=float, default=50.0)
    parser.add_argument("--pixel-pitch-mm", type=float, default=0.1)
    parser.add_argument("--beam-radius-mm", type=float, default=0.1)
    parser.add_argument("--focus-mm", type=float, default=320.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metal", type=Path, default=HERE / "multi_transport")
    parser.add_argument("--cumetal", type=Path, default=HERE / "cumetal_transport")
    parser.add_argument(
        "--cumetal-kernel", type=Path, default=HERE / "xrd_transport.cumetal.metal"
    )
    args = parser.parse_args()
    if not 0 < args.photons <= 0xFFFFFFFF or not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("photons and seed must fit UInt32; photons must be positive")

    geometry = [
        args.thickness_mm,
        args.gap_mm,
        args.detector_half_mm,
        args.pixel_pitch_mm,
        args.beam_radius_mm,
        args.focus_mm,
    ]
    scales = [1.0, 1.0, 1.0, 1.0]
    with tempfile.TemporaryDirectory(prefix="xrd_cuda_metal_") as temporary:
        temp = Path(temporary)
        bundle = temp / "input.bundle"
        metal_image = temp / "metal.raw"
        cumetal_image = temp / "cumetal.raw"
        subprocess.run(
            [
                sys.executable,
                str(HERE / "prepare_cuda_input.py"),
                *(f"{value:.12g}" for value in geometry + scales),
                str(args.physics),
                str(args.form_factor),
                str(bundle),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        metal = command_json(
            [
                str(args.metal),
                str(args.photons),
                *(f"{value:.12g}" for value in geometry + scales),
                str(args.seed),
                str(args.physics),
                str(args.form_factor),
                str(metal_image),
            ]
        )
        cumetal = command_json(
            [
                str(args.cumetal),
                str(args.cumetal_kernel),
                str(args.photons),
                str(args.seed),
                str(bundle),
                str(cumetal_image),
            ]
        )
        native_pixels = load_u32(metal_image)
        translated_pixels = load_u32(cumetal_image)
    if len(native_pixels) != len(translated_pixels):
        raise RuntimeError("Metal and CUDA detector-image dimensions differ")
    absolute_differences = [
        abs(left - right)
        for left, right in zip(native_pixels, translated_pixels)
        if left != right
    ]
    record = {
        "absolute_count_difference": sum(absolute_differences),
        "class_difference": [
            int(right) - int(left)
            for left, right in zip(metal["classes"], cumetal["classes"])
        ],
        "cuda_cumetal": cumetal,
        "detector_l1_fraction": (
            sum(absolute_differences) / metal["counts_on_detector"]
            if metal["counts_on_detector"]
            else 0.0
        ),
        "differing_pixels": len(absolute_differences),
        "exact_image_equal": native_pixels == translated_pixels,
        "form_factor": display_path(args.form_factor),
        "geometry": {
            "beam_radius_mm": args.beam_radius_mm,
            "detector_half_mm": args.detector_half_mm,
            "focus_from_entry_mm": args.focus_mm,
            "gap_mm": args.gap_mm,
            "pixel_pitch_mm": args.pixel_pitch_mm,
            "thickness_mm": args.thickness_mm,
        },
        "kernel_time_ratio_cumetal_over_metal": (
            cumetal["seconds_gpu_kernel"] / metal["seconds_gpu_kernel"]
        ),
        "kernel_timing_note": (
            "CuMetal uses synchronized host wall time after an MSL warm-up; "
            "native Metal uses command-buffer GPU timestamps"
        ),
        "max_pixel_difference": max(absolute_differences, default=0),
        "native_metal": metal,
        "photons": args.photons,
        "physics": display_path(args.physics),
        "seed": args.seed,
    }
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
