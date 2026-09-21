#!/usr/bin/env python3
"""Prepare one validated binary input shared by native CUDA and CuMetal.

The tables use the same six energy nodes and angular-CDF construction as the
Swift Metal host. Keeping preparation outside the C++ launcher also avoids a
platform-specific JSON dependency on NVIDIA/Windows systems.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Callable, Sequence

ENERGY_NODES = 6
RAYLEIGH_BINS = 4096


def read_columns(path: Path, *, skip_header: bool = False) -> tuple[list[float], list[float]]:
    lines = path.read_text().splitlines()[1 if skip_header else 0 :]
    rows = [line.split() for line in lines if line.strip()]
    pairs = [(float(row[0]), float(row[1])) for row in rows if len(row) >= 2]
    if len(pairs) < 2 or any(a[0] >= b[0] for a, b in zip(pairs, pairs[1:])):
        raise ValueError(f"Invalid form-factor table: {path}")
    return [row[0] for row in pairs], [row[1] for row in pairs]


def interpolate(value: float, grid: Sequence[float], samples: Sequence[float]) -> float:
    if value <= grid[0]:
        return samples[0]
    if value >= grid[-1]:
        return samples[-1]
    lower, upper = 0, len(grid) - 1
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if grid[middle] <= value:
            lower = middle
        else:
            upper = middle
    fraction = (value - grid[lower]) / (grid[upper] - grid[lower])
    return samples[lower] * (1 - fraction) + samples[upper] * fraction


def normalized_cdf(
    energies: Sequence[float], intensity: Callable[[float], float], label: str
) -> list[float]:
    tables: list[float] = []
    for energy in energies:
        cdf = [0.0] * (RAYLEIGH_BINS + 1)
        for index in range(1, RAYLEIGH_BINS + 1):
            theta = (index - 0.5) * math.pi / RAYLEIGH_BINS
            q = 2 * energy / 510.99895 * math.sin(theta / 2)
            cosine = math.cos(theta)
            cdf[index] = cdf[index - 1] + max(
                0.0, intensity(q) * (1 + cosine * cosine) * math.sin(theta)
            )
        if cdf[-1] <= 0:
            raise ValueError(f"Empty {label} Rayleigh distribution")
        total = cdf[-1]
        tables.extend(value / total for value in cdf)
    return tables


def flattened_shells(value: object, label: str) -> tuple[list[float], int]:
    if value is None:
        return [0.0], 0
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError(f"Invalid {label} Penelope oscillator table")
    flat: list[float] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or float(row[0]) <= 0
            or float(row[1]) < 0
            or float(row[2]) <= 0
            or not all(math.isfinite(float(item)) for item in row)
        ):
            raise ValueError(f"Invalid {label} Penelope oscillator table")
        flat.extend(float(item) for item in row)
    return flat, len(value)


def prepare(args: argparse.Namespace) -> dict[str, int | str]:
    physics = json.loads(args.physics.read_text())
    energies = [float(value) for value in physics["energy_kev"]]
    sample_xs = [[float(value) for value in row] for row in physics["sample_xs_mm_inv"]]
    air_xs = [[float(value) for value in row] for row in physics["air_xs_mm_inv"]]
    compton_cdf = [float(value) for value in physics["compton_cdf"]]
    if (
        len(energies) != ENERGY_NODES
        or any(a >= b for a, b in zip(energies, energies[1:]))
        or len(sample_xs) != 3
        or len(air_xs) != 3
        or any(len(row) != ENERGY_NODES for row in sample_xs + air_xs)
        or len(compton_cdf) != 1025
    ):
        raise ValueError("Invalid six-node physics table")

    sample_q, sample_ff = read_columns(args.form_factor)
    sample_rayleigh = normalized_cdf(
        energies,
        lambda q: interpolate(q, sample_q, sample_ff) ** 2,
        "sample",
    )

    data_root = os.environ.get("G4LEDATA")
    if not data_root:
        raise ValueError("G4LEDATA is required for air Rayleigh tables")
    air_dir = Path(data_root) / "penelope" / "rayleigh"
    nitrogen_q, nitrogen_ff = read_columns(air_dir / "pdaff07.p08", skip_header=True)
    oxygen_q, oxygen_ff = read_columns(air_dir / "pdaff08.p08", skip_header=True)
    oxygen_to_nitrogen = (0.3 / 15.9994) / (0.7 / 14.0067)

    def air_intensity(q: float) -> float:
        nitrogen = interpolate(q, nitrogen_q, nitrogen_ff)
        oxygen = interpolate(q, oxygen_q, oxygen_ff)
        return nitrogen * nitrogen + oxygen_to_nitrogen * oxygen * oxygen

    air_rayleigh = normalized_cdf(energies, air_intensity, "air")
    sample_shells, sample_shell_count = flattened_shells(
        physics.get("sample_compton_shells"), "sample"
    )
    air_shells, air_shell_count = flattened_shells(
        physics.get("air_compton_shells"), "air"
    )

    sample_lateral = float(physics.get("sample_lateral_mm", 120.0))
    if sample_lateral <= 0:
        raise ValueError("sample_lateral_mm must be positive")
    pixels_float = 2 * args.detector_half_mm / args.pixel_pitch_mm
    pixels = round(pixels_float)
    if pixels <= 0 or pixels * pixels >= 4_194_304 or abs(pixels_float - pixels) >= 1e-4:
        raise ValueError("Detector width must be an exact supported number of pixels")
    if args.focus_from_entry_mm <= args.thickness_mm + args.gap_mm:
        raise ValueError("Focus must lie beyond the detector plane")

    parameters = [
        args.thickness_mm,
        args.gap_mm,
        args.detector_half_mm,
        args.pixel_pitch_mm,
        args.beam_radius_mm,
        args.focus_from_entry_mm,
        args.density_scale,
        args.photo_scale,
        args.compton_scale,
        args.rayleigh_scale,
        float(physics.get("source_energy_kev", 22.162917)),
        float(physics.get("upstream_air_mm", 0.0)),
        float(sample_shell_count),
        float(air_shell_count),
        sample_lateral * 0.5,
    ]
    cross_sections = energies + [value for row in sample_xs + air_xs for value in row]
    arrays = [
        cross_sections,
        sample_rayleigh,
        compton_cdf,
        parameters,
        air_rayleigh,
        sample_shells,
        air_shells,
    ]
    header = struct.pack(
        "<8s9I",
        b"XRDCU01\0",
        1,
        pixels,
        *(len(values) for values in arrays),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(header)
        for values in arrays:
            stream.write(struct.pack(f"<{len(values)}f", *values))
    return {
        "air_shells": air_shell_count,
        "bytes": args.output.stat().st_size,
        "output": str(args.output),
        "pixels": pixels,
        "sample_shells": sample_shell_count,
    }


def positive(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thickness_mm", type=positive)
    parser.add_argument("gap_mm", type=nonnegative)
    parser.add_argument("detector_half_mm", type=positive)
    parser.add_argument("pixel_pitch_mm", type=positive)
    parser.add_argument("beam_radius_mm", type=nonnegative)
    parser.add_argument("focus_from_entry_mm", type=positive)
    parser.add_argument("density_scale", type=positive)
    parser.add_argument("photo_scale", type=nonnegative)
    parser.add_argument("compton_scale", type=nonnegative)
    parser.add_argument("rayleigh_scale", type=nonnegative)
    parser.add_argument("physics", type=Path)
    parser.add_argument("form_factor", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), sort_keys=True))
