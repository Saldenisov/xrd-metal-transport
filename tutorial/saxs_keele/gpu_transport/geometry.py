"""Shared geometry description for matched Geant4 and Metal slab runs.

Geant4 still constructs its own solids, and Metal still navigates its own box.
This object removes duplicated dimensions and emits inputs for both engines.
It does not translate arbitrary Geant4 geometry to the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class SlabDetectorGeometry:
    """Finite homogeneous box, circular source and square entrance detector."""

    sample_thickness_mm: float
    sample_lateral_mm: float
    downstream_air_mm: float
    detector_pixels: int
    pixel_pitch_mm: float
    beam_radius_mm: float
    focus_from_entry_mm: float | None
    upstream_air_mm: float = 0.1
    sample_center_z_mm: float = 100.0
    detector_thickness_mm: float = 0.3

    @classmethod
    def from_manifest(cls, manifest: dict) -> "SlabDetectorGeometry":
        """Build the supported geometry from an inverse-model manifest."""
        thickness = float(manifest["sample_thickness_mm"])
        upstream = float(manifest.get("upstream_air_mm", 0.1))
        sample_entry_z = float(manifest["source_plane_z_mm"]) + upstream
        return cls(
            sample_thickness_mm=thickness,
            sample_lateral_mm=float(manifest.get("sample_lateral_mm", 120.0)),
            downstream_air_mm=float(manifest["front_face_to_detector_mm"]) - thickness,
            detector_pixels=int(manifest["detector_pixels"]),
            pixel_pitch_mm=float(manifest["pixel_pitch_um"]) * 1e-3,
            beam_radius_mm=float(manifest["source_diameter_um"]) * 5e-4,
            focus_from_entry_mm=float(manifest["focus_point_z_mm"]) - sample_entry_z,
            upstream_air_mm=upstream,
        )

    def __post_init__(self) -> None:
        positive = (
            self.sample_thickness_mm,
            self.sample_lateral_mm,
            self.detector_pixels,
            self.pixel_pitch_mm,
            self.detector_thickness_mm,
        )
        if min(positive) <= 0 or min(
            self.downstream_air_mm, self.beam_radius_mm, self.upstream_air_mm
        ) < 0:
            raise ValueError("Geometry lengths and detector size are invalid")
        if self.focus_from_entry_mm is not None and (
            self.focus_from_entry_mm
            <= self.sample_thickness_mm + self.downstream_air_mm
        ):
            raise ValueError("Focus must lie beyond the detector plane")

    @property
    def detector_half_mm(self) -> float:
        return self.detector_pixels * self.pixel_pitch_mm / 2

    @property
    def front_face_to_detector_mm(self) -> float:
        return self.sample_thickness_mm + self.downstream_air_mm

    @property
    def mean_plane_to_detector_mm(self) -> float:
        return self.sample_thickness_mm / 2 + self.downstream_air_mm

    @property
    def sample_entry_z_mm(self) -> float:
        return self.sample_center_z_mm - self.sample_thickness_mm / 2

    def geant4_detector_commands(self) -> str:
        """Return pre-initialization detector-construction commands."""
        detector_size = 2 * self.detector_half_mm * sqrt(2)
        detector_distance = self.mean_plane_to_detector_mm + self.detector_thickness_mm / 2
        return f"""/det/setPhantomBox true
/det/setPhantomDiameter {self.sample_thickness_mm:.12g} mm
/det/setPhantomHeight {self.sample_lateral_mm:.12g} mm
/det/setPhantomZ {self.sample_center_z_mm:.12g} mm
/det/setSlits false
/det/setDetectorSize {detector_size:.9g} mm
/det/setDetectorThickness {self.detector_thickness_mm:.12g} mm
/det/setDetectorSampleDistance {detector_distance:.12g} mm"""

    def geant4_source_commands(self) -> str:
        """Return post-initialization GPS source commands."""
        source_z = self.sample_entry_z_mm - self.upstream_air_mm
        source_direction = "/gps/direction 0 0 1"
        if self.focus_from_entry_mm is not None:
            focus_z = self.sample_entry_z_mm + self.focus_from_entry_mm
            source_direction = (
                "/gps/ang/type focused\n"
                f"/gps/ang/focuspoint 0 0 {focus_z:.12g} mm"
            )
        return f"""/gps/pos/type Plane
/gps/pos/shape Circle
/gps/pos/radius {self.beam_radius_mm:.12g} mm
/gps/pos/centre 0 0 {source_z:.12g} mm
{source_direction}"""

    def metal_arguments(self) -> list[str]:
        """Return the geometric part of the Metal command line."""
        focus = self.focus_from_entry_mm if self.focus_from_entry_mm is not None else 1e9
        return [
            f"{self.sample_thickness_mm:.12g}",
            f"{self.downstream_air_mm:.12g}",
            f"{self.detector_half_mm:.12g}",
            f"{self.pixel_pitch_mm:.12g}",
            f"{self.beam_radius_mm:.12g}",
            f"{focus:.12g}",
        ]

    def manifest(self, *, energy_kev: float, q_range_nm_inv: tuple[float, float],
                 radial_points: int) -> dict[str, float | int | list[float]]:
        return {
            "detector_pixels": self.detector_pixels,
            "pixel_pitch_um": self.pixel_pitch_mm * 1000,
            "sample_thickness_mm": self.sample_thickness_mm,
            "sample_lateral_mm": self.sample_lateral_mm,
            "front_face_to_detector_mm": self.front_face_to_detector_mm,
            "mean_scattering_plane_to_detector_mm": self.mean_plane_to_detector_mm,
            "energy_kev": energy_kev,
            "q_range_nm_inv": list(q_range_nm_inv),
            "radial_points": radial_points,
        }
