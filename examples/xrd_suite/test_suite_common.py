"""Regression checks for the published XRD suite integration."""

from __future__ import annotations

import numpy as np

from suite_common import pyfai_operator


def test_radial_operator_discards_clipped_edge_bins() -> None:
    manifest = {
        "energy_kev": 22.162917,
        "detector_pixels": 1000,
        "pixel_pitch_um": 100.0,
        "mean_scattering_plane_to_detector_mm": 150.0,
        "radial_points": 256,
        "q_range_nm_inv": [1.0, 30.0],
    }
    q, weights = pyfai_operator(manifest)
    row_weight = np.asarray(weights.sum(axis=1)).ravel()

    assert q.shape == (256,)
    assert weights.shape == (256, 1_000_000)
    assert 0.7 < row_weight[0] / row_weight[1] < 1.1
    assert 0.95 < row_weight[-1] / row_weight[-2] < 1.05
