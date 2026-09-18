"""Apply one thickness-corrected XRD integration to a detector image."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xrd_preprocessing import AzimuthalIntegration

HC_KEV_NM = 1.2398419843320026


def integrate_image(image: np.ndarray, geometry: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Return q, count-space radial sums and corrected distance in mm."""
    pixels = int(geometry["detector_pixels"])
    frame = pd.DataFrame([{
        "measurement_data": image,
        "calculated_distance": geometry["front_face_to_detector_mm"] * 1e-3,
        "pixel_size": geometry["pixel_pitch_um"],
        "center": (pixels / 2.0, pixels / 2.0),
        "wavelength": HC_KEV_NM / geometry["energy_kev"],
        "sample_thickness_mm": geometry["sample_thickness_mm"],
        "interpolation_q_range": tuple(geometry["q_range_nm_inv"]),
    }])
    output = AzimuthalIntegration(
        column="measurement_data",
        output_column="radial_profile_data",
        q_range_column="q_range",
        npt=int(geometry["radial_points"]),
        mode="1D",
        calibration_mode="dataframe",
        error_model="poisson",
        correct_solid_angle=False,
        thickness_adjustment=True,
        require_thickness_adjustment=True,
        thickness_reference_mm=0.0,
        sample_thickness_column="sample_thickness_mm",
    ).fit_transform(frame).iloc[0]
    distance_mm = float(output["calculated_distance"]) * 1000.0
    if not np.isclose(distance_mm, geometry["mean_scattering_plane_to_detector_mm"], atol=1e-6):
        raise AssertionError("XRD-preprocessing thickness correction mismatch")
    q = np.asarray(output["q_range"], dtype=float)
    counts = np.asarray(output["radial_sum_signal"], dtype=float)
    if counts.shape != q.shape:
        raise AssertionError("XRD-preprocessing did not return 1D ring sums")
    return q, counts, distance_mm

