"""Fast checks for the long-running independent-history parity benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_water_100mm_cpu_metal import (  # noqa: E402
    MANIFEST,
    geant4_macro,
    metal_seed,
    profile_parity,
    pyfai_operator,
    read_g4_channels,
)
from validate_multi_transport import (  # noqa: E402
    integrate as integrate_breast,
    pyfai_operator as breast_pyfai_operator,
)


def test_geometry_and_parallel_source() -> None:
    assert MANIFEST["mean_scattering_plane_to_detector_mm"] == 150.0
    assert MANIFEST["front_face_to_detector_mm"] == 200.0
    assert MANIFEST["sample_lateral_mm"] == 120.0
    macro = geant4_macro(Path("/tmp/water_unit_test"), 1000, 0)
    assert "/det/setPhantomDiameter 100 mm" in macro
    assert "/det/setPhantomHeight 120 mm" in macro
    assert "/det/setDetectorSampleDistance 150.15 mm" in macro
    assert "/gps/pos/radius 0.05 mm" in macro
    assert "/gps/direction 0 0 1" in macro
    assert "/random/setSeeds 600031 700033" in macro
    assert "/random/setSeeds 702131 803133" in geant4_macro(
        Path("/tmp/water_unit_test"), 1000, 100)


def test_channel_histogram_parser(tmp_path: Path) -> None:
    path = tmp_path / "channels.csv"
    path.write_text("#class tools::histo::h1d\nentries,center\n0,-1\n10,0\n20,1\n30,2\n40,3\n0,4\n")
    np.testing.assert_array_equal(read_g4_channels(path), [10, 20, 30, 40])


def test_independent_pixel_covariance() -> None:
    weight = csr_matrix([[1.0, 0.5, 0.0, 0.0], [0.0, 0.5, 1.0, 1.0]])
    image = np.asarray([[5, 10], [7, 3]], dtype=np.int64)
    result = profile_parity(image, image, np.asarray([1.0, 2.0]), weight)
    assert result["covariance_rank"] == 2
    assert result["chi2"] == 0
    assert result["chi2_p_value"] == 1
    np.testing.assert_array_equal(result["cpu_profile"], result["gpu_profile"])


def test_pyfai_operator_has_exact_requested_geometry() -> None:
    q, weight = pyfai_operator()
    assert q.shape == (200,)
    assert 1.0 < q[0] < 1.2
    assert 29.8 < q[-1] < 30.0
    assert weight.shape == (200, 1_000_000)
    assert weight.nnz > 200


def test_breast_xrd_preprocessing_matches_csr_operator() -> None:
    image = np.zeros((1000, 1000), dtype=np.int32)
    image[420, 510] = 23
    image[530, 460] = 17
    q, intensity = integrate_breast(image)
    q_operator, weights = breast_pyfai_operator()
    np.testing.assert_array_equal(q, q_operator)
    np.testing.assert_allclose(intensity, weights @ image.ravel(), rtol=1e-9, atol=1e-7)


def test_metal_repetitions_have_distinct_philox_keys_and_counters() -> None:
    photons, repetitions, blocks = 200, 10, 4
    keys = [metal_seed(photons, run) for run in range(repetitions)]
    assert len(set(keys)) == repetitions
    inputs = {(keys[run], history, block)
              for run in range(repetitions)
              for history in range(photons)
              for block in range(blocks)}
    assert len(inputs) == repetitions * photons * blocks
