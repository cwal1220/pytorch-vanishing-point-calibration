from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PitchYawResult:
    pitch: float
    yaw: float
    delta_pitch: float
    delta_yaw: float
    focal_length_px: float
    original_vp_x: float
    original_vp_y: float
    predicted_vp_x: float
    predicted_vp_y: float
    unit: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def scale_focal_length_px(
    original_focal_length_px: float,
    original_width: int,
    resized_width: int,
) -> float:
    return float(original_focal_length_px) * float(resized_width) / float(original_width)


def compute_pitch_yaw_from_vp(
    predicted_vp_x: float,
    predicted_vp_y: float,
    original_vp_x: float,
    original_vp_y: float,
    focal_length_px: float,
    original_pitch: float = 0.0,
    original_yaw: float = 0.0,
    angles_in_degrees: bool = False,
) -> PitchYawResult:
    """Implements the paper's equations (1)-(3).

    The paper defines:
      pitch = original_pitch + delta_pitch
      delta_pitch = (original_vp_y - vp_y) / focal_length
    and states yaw uses the analogous x-coordinate equation.
    """

    delta_pitch = (float(original_vp_y) - float(predicted_vp_y)) / float(focal_length_px)
    delta_yaw = (float(original_vp_x) - float(predicted_vp_x)) / float(focal_length_px)

    pitch = float(original_pitch) + delta_pitch
    yaw = float(original_yaw) + delta_yaw
    unit = "radians"

    if angles_in_degrees:
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        delta_pitch = math.degrees(delta_pitch)
        delta_yaw = math.degrees(delta_yaw)
        unit = "degrees"

    return PitchYawResult(
        pitch=pitch,
        yaw=yaw,
        delta_pitch=delta_pitch,
        delta_yaw=delta_yaw,
        focal_length_px=float(focal_length_px),
        original_vp_x=float(original_vp_x),
        original_vp_y=float(original_vp_y),
        predicted_vp_x=float(predicted_vp_x),
        predicted_vp_y=float(predicted_vp_y),
        unit=unit,
    )


def rotation_matrix_from_pitch_yaw_roll(
    pitch: float,
    yaw: float,
    roll: float = 0.0,
) -> np.ndarray:
    """Build a rotation matrix from pitch/yaw/roll in radians.

    This is a practical implementation helper for downstream projection logic.
    Axis convention:
      - pitch: rotation about x
      - yaw: rotation about y
      - roll: rotation about z
    """

    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def project_forward_axis_to_vp(
    focal_length_px: float,
    image_width: int,
    image_height: int,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    principal_point_x: float | None = None,
    principal_point_y: float | None = None,
) -> tuple[float, float]:
    """Project a far-away point on the world forward axis to image space.

    This is an implementation inference based on the paper's description of
    projecting a point far along the Z axis. It is useful for constructing the
    original/theoretical VP used in the pitch/yaw update equations.
    """

    cx = float(principal_point_x) if principal_point_x is not None else (image_width - 1) * 0.5
    cy = float(principal_point_y) if principal_point_y is not None else (image_height - 1) * 0.5

    rotation = rotation_matrix_from_pitch_yaw_roll(pitch=pitch, yaw=yaw, roll=roll)
    world_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_direction = rotation @ world_forward

    if abs(camera_direction[2]) < 1e-8:
        raise ValueError("Forward axis projects to infinity because z is too close to zero.")

    x = cx + focal_length_px * camera_direction[0] / camera_direction[2]
    y = cy + focal_length_px * camera_direction[1] / camera_direction[2]
    return float(x), float(y)
