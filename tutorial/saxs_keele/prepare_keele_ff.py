#!/usr/bin/env python3
"""Convert Keele I(q)=F(q)^2/W tables to Geant4 MI form-factor files.

Geant4's G4PenelopeRayleighModelMI expects two whitespace-separated columns:
dimensionless q/(m_e c) and F(q)/sqrt(W).  Keele's xrd_components.txt stores
q in nm^-1 and I(q)=F(q)^2/W in six tab-separated columns.
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
from pathlib import Path


# Reduced electron Compton wavelength, hbar/(m_e c), in nm (CODATA 2022).
ELECTRON_REDUCED_COMPTON_NM = 3.8615926744e-4

COMPONENTS = {
    "fat": {
        "columns": (0, 1),
        "reference": "FF_fat_Tartari2002_joint_lowXdata_ESRF2003.dat",
        "density": 0.923,
        "elements": {"H": 0.1190, "C": 0.7720, "N": 0.0, "O": 0.1090},
    },
    "collagen": {
        "columns": (2, 3),
        "reference": "FF_bonematrix_Tartari2002.dat",
        "density": 1.263,
        "elements": {"H": 0.0344, "C": 0.7140, "N": 0.1827, "O": 0.0689},
    },
    "water": {
        "columns": (4, 5),
        "reference": "FF_water_Tartari2002.dat",
        "density": 1.000,
        "elements": {"H": 0.1119, "C": 0.0, "N": 0.0, "O": 0.8881},
    },
}


def read_components(path: Path) -> dict[str, list[tuple[float, float]]]:
    rows = {name: [] for name in COMPONENTS}
    with path.open(encoding="utf-8") as stream:
        next(stream)
        for line_number, line in enumerate(stream, start=2):
            fields = line.rstrip("\n").split("\t")
            fields.extend([""] * (6 - len(fields)))
            for name, metadata in COMPONENTS.items():
                q_column, intensity_column = metadata["columns"]
                if not fields[q_column].strip() or not fields[intensity_column].strip():
                    continue
                try:
                    q_nm = float(fields[q_column])
                    intensity = float(fields[intensity_column])
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value at {path}:{line_number}") from exc
                if q_nm <= 0 or intensity < 0 or not math.isfinite(q_nm + intensity):
                    raise ValueError(f"Non-physical value at {path}:{line_number}")
                rows[name].append((q_nm, intensity))

    for name, values in rows.items():
        values.sort()
        if len(values) < 2:
            raise ValueError(f"Component {name} has fewer than two valid points")
        if any(right[0] <= left[0] for left, right in zip(values, values[1:])):
            raise ValueError(f"Component {name} q values are not strictly increasing")
    return rows


def interpolate(points: list[tuple[float, float]], q_nm: float) -> float:
    q_values = [point[0] for point in points]
    index = bisect.bisect_left(q_values, q_nm)
    if index == 0:
        return points[0][1]
    if index == len(points):
        return points[-1][1]
    q0, y0 = points[index - 1]
    q1, y1 = points[index]
    fraction = (q_nm - q0) / (q1 - q0)
    return y0 + fraction * (y1 - y0)


def read_reference_miff(directory: Path) -> dict[str, list[tuple[float, float]]]:
    references = {}
    for name, metadata in COMPONENTS.items():
        path = directory / str(metadata["reference"])
        points = []
        with path.open(encoding="ascii") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    q_geant4, form_factor = map(float, line.split())
                except ValueError as exc:
                    raise ValueError(f"Invalid Geant4 MIFF row at {path}:{line_number}") from exc
                points.append(
                    (q_geant4 / ELECTRON_REDUCED_COMPTON_NM, form_factor * form_factor)
                )
        references[name] = points
    return references


def stitch_reference_tails(
    measured: list[tuple[float, float]], reference: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    q_min, q_max = measured[0][0], measured[-1][0]
    return (
        [point for point in reference if point[0] < q_min]
        + measured
        + [point for point in reference if point[0] > q_max]
    )


def write_geant4_ff(path: Path, points: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        for q_nm, intensity in points:
            q_geant4 = q_nm * ELECTRON_REDUCED_COMPTON_NM
            form_factor = math.sqrt(intensity)
            stream.write(f"{q_geant4:.12e} {form_factor:.12e}\n")


def mixture_points(
    rows: dict[str, list[tuple[float, float]]], fractions: dict[str, float]
) -> list[tuple[float, float]]:
    q_min = max(points[0][0] for points in rows.values())
    q_max = min(points[-1][0] for points in rows.values())
    q_grid = sorted(
        {
            q_nm
            for points in rows.values()
            for q_nm, _ in points
            if q_min <= q_nm <= q_max
        }
    )
    return [
        (
            q_nm,
            sum(
                fractions[name] * interpolate(rows[name], q_nm)
                for name in COMPONENTS
            ),
        )
        for q_nm in q_grid
    ]


def mixture_material(fractions: dict[str, float]) -> tuple[float, dict[str, float]]:
    inverse_density = sum(
        fractions[name] / float(COMPONENTS[name]["density"]) for name in COMPONENTS
    )
    density = 1.0 / inverse_density
    elements = {
        element: sum(
            fractions[name] * float(COMPONENTS[name]["elements"][element])
            for name in COMPONENTS
        )
        for element in ("H", "C", "N", "O")
    }
    return density, elements


def write_macro(
    path: Path,
    ff_path_from_build: str,
    fractions: dict[str, float],
    events: int,
    output_name: str,
) -> None:
    density, elements = mixture_material(fractions)
    macro = f"""# Generated Keele fat/water/collagen mixture.
# Mass fractions: fat={fractions['fat']:.6f}, water={fractions['water']:.6f}, collagen={fractions['collagen']:.6f}
/det/setCustomMatDensity {density:.10f}
/det/setCustomMatHmassfract {elements['H']:.10f}
/det/setCustomMatCmassfract {elements['C']:.10f}
/det/setCustomMatNmassfract {elements['N']:.10f}
/det/setCustomMatOmassfract {elements['O']:.10f}
/det/SetCustomMatFF {ff_path_from_build}

/det/setPhantomMaterial 30
/det/setPhantomDiameter 10. mm
/det/setPhantomHeight 1. mm
/det/setPhantomZ 500. mm
/det/setSlits false
/det/setDetectorSize 300. mm
/det/setDetectorThickness 1. mm
/det/setDetectorSampleDistance 500. mm

/phys/SelectPhysicsList empenelopeMI
/run/setCut 0.1 mm
/run/verbose 1
/run/initialize
/run/setfilenamesave results/{output_name}

/gps/pos/type Plane
/gps/pos/shape Circle
/gps/pos/radius 0.1 mm
/gps/pos/centre 0. 0. 0. mm
/gps/direction 0 0 1
/gps/particle gamma
/gps/ene/type Mono
/gps/ene/mono 20. keV

/run/printProgress 10000
/run/beamOn {events}
"""
    path.write_text(macro, encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "xrd_components.txt",
        help="Keele six-column xrd_components.txt",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data")
    led = os.environ.get("G4LEDATA")
    default_miff_dir = Path(led) / "penelope" / "rayleigh" / "MIFF" if led else None
    parser.add_argument(
        "--geant4-miff-dir",
        type=Path,
        default=default_miff_dir,
        help="Reference MIFF directory used for physically defined low/high-q tails",
    )
    parser.add_argument(
        "--fractions",
        nargs=3,
        type=float,
        metavar=("FAT", "WATER", "COLLAGEN"),
        help="Also generate one mixture form factor and run macro",
    )
    parser.add_argument("--name", default="keele_mix")
    parser.add_argument("--events", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.geant4_miff_dir is None:
        raise ValueError("Set G4LEDATA or pass --geant4-miff-dir")
    measured_rows = read_components(args.input.resolve())
    reference_rows = read_reference_miff(args.geant4_miff_dir.resolve())
    rows = {
        name: stitch_reference_tails(measured_rows[name], reference_rows[name])
        for name in COMPONENTS
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, points in rows.items():
        output = args.output_dir / f"keele_{name}.dat"
        write_geant4_ff(output, points)
        measured = measured_rows[name]
        print(
            f"{name}: {len(measured)} Keele points, "
            f"q={measured[0][0]:.5g}..{measured[-1][0]:.5g} nm^-1; "
            f"reference tails -> {output}"
        )

    if args.fractions is None:
        return

    if args.events <= 0:
        raise ValueError("--events must be positive")
    fractions = dict(zip(("fat", "water", "collagen"), args.fractions))
    if any(value < 0 or value > 1 for value in fractions.values()):
        raise ValueError("Each mass fraction must be in [0, 1]")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Mass fractions must sum to 1")

    mixed = mixture_points(rows, fractions)
    ff_output = args.output_dir / f"{args.name}.dat"
    macro_output = Path(f"{args.name}.mac")
    write_geant4_ff(ff_output, mixed)
    write_macro(macro_output, f"data/{args.name}.dat", fractions, args.events, args.name)
    print(f"mixture: {len(mixed)} points -> {ff_output}")
    print(f"macro: {macro_output}")


if __name__ == "__main__":
    main()
