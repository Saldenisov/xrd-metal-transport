"""Fast Metal branch tests; high-statistical Geant4 parity is separate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
METAL = HERE / "multi_transport"
PHYSICS = HERE / "results" / "multi_transport" / "water_22p022_physics.json"
SHELL_PHYSICS = HERE / "results" / "multi_transport" / "water_22p022_shell_exact_xs_probe.json"
MIFF = HERE.parent / "data" / "keele_water.dat"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from diagnose_transport_ablation import (DEFAULT_PHYSICS, run_geant4 as run_g4_ablation,
                                           run_metal as run_metal_ablation)  # noqa: E402
from metal_forward import BASELINE_FRACTIONS, PHYSICS_30, composition_compton_shells  # noqa: E402
from validate_water_100mm_cpu_metal import G4  # noqa: E402


def run_channel_case(tmp_path: Path, *, phot: float, compt: float,
                     rayl: float, air_rayl: float = 0.0,
                     air_compt: float = 0.0, photons: int = 200_000,
                     seed: int = 900_007, thickness_mm: float = 10.0,
                     sample_lateral_mm: float = 120.0) -> dict:
    physics = json.loads(PHYSICS.read_text())
    physics["upstream_air_mm"] = 0.0
    physics["sample_xs_mm_inv"] = [[phot] * 6, [compt] * 6, [rayl] * 6]
    physics["air_xs_mm_inv"] = [[0.0] * 6, [air_compt] * 6, [air_rayl] * 6]
    physics["sample_lateral_mm"] = sample_lateral_mm
    table = tmp_path / "physics.json"
    table.write_text(json.dumps(physics))
    raw = tmp_path / "image.raw"
    command = [str(METAL), str(photons), str(thickness_mm), "100", "50", "0.1", "0.05",
               "1000000000", "1", "1", "1", "1", str(seed), str(table),
               str(MIFF), str(raw)]
    result = json.loads(subprocess.check_output(command, text=True))
    image = np.fromfile(raw, dtype=np.uint32)
    assert int(image.sum()) == result["counts_on_detector"]
    assert result["interaction_cap_count"] == 0
    return result


def test_direct_channel_without_interactions(tmp_path: Path) -> None:
    result = run_channel_case(tmp_path, phot=0.0, compt=0.0, rayl=0.0)
    assert result["classes"][0] == result["photons"]
    assert sum(result["classes"][1:]) == 0


def test_large_virtual_detector_keeps_full_pixel_index(tmp_path: Path) -> None:
    physics = json.loads(PHYSICS.read_text())
    physics["sample_xs_mm_inv"] = [[0.0] * 6 for _ in range(3)]
    physics["air_xs_mm_inv"] = [[0.0] * 6 for _ in range(3)]
    table = tmp_path / "physics.json"
    table.write_text(json.dumps(physics))
    raw = tmp_path / "large_image.raw"
    pixels = 1536
    pitch_mm = 0.1353
    command = [
        str(METAL), "10000", "2", "498", str(pixels * pitch_mm / 2),
        str(pitch_mm), "0.05", "1000000000", "1", "1", "1", "1",
        "987654", str(table), str(MIFF), str(raw),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    image = np.fromfile(raw, dtype=np.uint32).reshape(pixels, pixels)
    assert result["detector_pixels"] == pixels
    assert result["classes"][0] == 10000
    assert int(image.sum()) == 10000
    assert int(image[pixels // 2 - 1:pixels // 2 + 2,
                     pixels // 2 - 1:pixels // 2 + 2].sum()) == 10000


def test_photoelectric_attenuation(tmp_path: Path) -> None:
    result = run_channel_case(tmp_path, phot=0.02, compt=0.0, rayl=0.0)
    direct = result["classes"][0]
    expected = result["photons"] * np.exp(-0.02 * 10.0)
    sigma = np.sqrt(result["photons"] * np.exp(-0.2) * (1 - np.exp(-0.2)))
    assert abs(direct - expected) < 5 * sigma
    assert result["classes"][5] + direct == result["photons"]


def test_repeated_rayleigh_branch(tmp_path: Path) -> None:
    result = run_channel_case(tmp_path, phot=0.0, compt=0.0, rayl=0.05)
    assert result["classes"][1] > 0
    assert result["classes"][2] > 0
    assert result["classes"][3] == 0
    assert result["classes"][5] == 0


def test_repeated_compton_branch(tmp_path: Path) -> None:
    result = run_channel_case(tmp_path, phot=0.0, compt=0.05, rayl=0.0)
    assert result["classes"][3] > 0
    assert result["classes"][1] == 0
    assert result["classes"][2] == 0
    assert result["classes"][5] == 0


def test_finite_sample_side_exits_change_thick_compton_transport(tmp_path: Path) -> None:
    arguments = dict(phot=0.0, compt=0.018, rayl=0.0, photons=300_000,
                     thickness_mm=100.0)
    finite = run_channel_case(tmp_path, **arguments, sample_lateral_mm=120.0)
    effectively_infinite = run_channel_case(tmp_path, **arguments,
                                            sample_lateral_mm=10_000.0)
    assert effectively_infinite["classes"][3] > 1.1 * finite["classes"][3]
    assert abs(effectively_infinite["classes"][0] - finite["classes"][0]) < 500


@pytest.mark.skipif(not G4.exists(), reason="Geant4 saxs binary is not built")
def test_transport_only_matches_geant4_detector_geometry(tmp_path: Path) -> None:
    cpu = run_g4_ablation(tmp_path, 100_000, 8, "transport_only")
    gpu = run_metal_ablation(tmp_path, 100_000, DEFAULT_PHYSICS,
                             "transport_only", 925_103)
    assert cpu.tolist() == gpu.tolist() == [100_000, 0, 0, 0]


@pytest.mark.skipif(not G4.exists(), reason="Geant4 saxs binary is not built")
def test_finite_box_compton_ablation_matches_geant4(tmp_path: Path) -> None:
    cpu = run_g4_ablation(tmp_path, 1_000_000, 8, "compton_only")
    gpu = run_metal_ablation(tmp_path, 1_000_000, DEFAULT_PHYSICS,
                             "compton_only", 925_103)
    assert abs(gpu[0] / cpu[0] - 1) < 0.02
    assert abs(gpu[3] / cpu[3] - 1) < 0.05


def test_air_rayleigh_changes_transport_but_not_sample_channel(tmp_path: Path) -> None:
    result = run_channel_case(tmp_path, phot=0.0, compt=0.0, rayl=0.0,
                              air_rayl=0.0001)
    assert result["air_interaction_hits"] > 0
    assert result["classes"][0] > 0
    assert result["classes"][1] + result["classes"][2] == 0
    assert result["classes"][3] == 0


def test_air_compton_changes_transport_but_not_sample_channel(tmp_path: Path) -> None:
    result = run_channel_case(tmp_path, phot=0.0, compt=0.0, rayl=0.0,
                              air_compt=0.0001)
    assert result["air_interaction_hits"] > 0
    assert result["classes"][0] > 0
    assert result["classes"][3] == 0
    assert result["classes"][1] + result["classes"][2] == 0


def test_philox_histories_reproduce_with_same_key(tmp_path: Path) -> None:
    kwargs = dict(phot=0.02, compt=0.03, rayl=0.02, photons=500_000)
    first = run_channel_case(tmp_path, **kwargs)
    first_image = np.fromfile(tmp_path / "image.raw", dtype=np.uint32).copy()
    second = run_channel_case(tmp_path, **kwargs)
    second_image = np.fromfile(tmp_path / "image.raw", dtype=np.uint32)
    assert first["classes"] == second["classes"]
    assert np.array_equal(first_image, second_image)
    third = run_channel_case(tmp_path, **kwargs, seed=900_019)
    third_image = np.fromfile(tmp_path / "image.raw", dtype=np.uint32)
    assert first["classes"] != third["classes"]
    assert not np.array_equal(first_image, third_image)


def test_philox_metal_known_answers(tmp_path: Path) -> None:
    binary = tmp_path / "philox_metal_kat"
    subprocess.run(["swiftc", "-O", "-framework", "Metal",
                    str(HERE / "philox_metal_kat.swift"), "-o", str(binary)], check=True)
    message = subprocess.check_output([str(binary)], text=True)
    assert "3/3 Random123 known-answer vectors passed" in message


def test_penelope_shell_compton_has_doppler_energy_broadening(tmp_path: Path) -> None:
    output = tmp_path / "compton.raw"
    physics = json.loads(SHELL_PHYSICS.read_text())
    assert len(physics["sample_compton_shells"]) == 2
    assert len(physics["air_compton_shells"]) == 3
    subprocess.run([str(METAL), "--probe-compton", "100000", "123",
                    str(physics["source_energy_kev"]), str(SHELL_PHYSICS),
                    "sample", str(output)], check=True, capture_output=True)
    outcomes = np.fromfile(output, dtype="<f4").reshape(-1, 2)
    assert np.isfinite(outcomes).all()
    assert np.all((outcomes[:, 0] >= 0) & (outcomes[:, 0] <= np.pi))
    assert np.all(outcomes[:, 1] > 0)
    energy = physics["source_energy_kev"]
    free_electron_energy = energy / (1 + energy / 510.99895 *
                                     (1 - np.cos(outcomes[:, 0])))
    assert np.std(outcomes[:, 1] - free_electron_energy) > 0.1


def test_penelope_air_shell_compton_probe(tmp_path: Path) -> None:
    output = tmp_path / "air_compton.raw"
    physics = json.loads(SHELL_PHYSICS.read_text())
    subprocess.run([str(METAL), "--probe-compton", "100000", "124",
                    str(physics["source_energy_kev"]), str(SHELL_PHYSICS),
                    "air", str(output)], check=True, capture_output=True)
    outcomes = np.fromfile(output, dtype="<f4").reshape(-1, 2)
    assert np.isfinite(outcomes).all()
    assert np.all(outcomes[:, 0] >= 0)
    assert np.all(outcomes[:, 1] > 0)


def test_trial_material_gets_its_own_geant4_oscillators() -> None:
    baseline = composition_compton_shells(*(BASELINE_FRACTIONS[name]
                                           for name in ("water", "fat", "collagen")))
    reference = json.loads((HERE / "results" / "multi_transport" /
                            "g50_22p163_shell_joint.json").read_text())
    np.testing.assert_allclose(baseline, reference["sample_compton_shells"], rtol=0, atol=0)
    trial = composition_compton_shells(0.45, 0.45, 0.10)
    assert len(trial) == len(baseline)
    assert not np.allclose(trial, baseline)


def test_30kev_cross_section_grid_is_strictly_increasing() -> None:
    physics = json.loads(PHYSICS_30.read_text())
    energies = np.asarray(physics["energy_kev"])
    assert np.all(np.diff(energies) > 0)
    assert energies[-1] == physics["source_energy_kev"] == 30.0
