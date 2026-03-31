from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LaneFit:
    polynomial_degree: int
    coefficients: tuple[float, ...]
    tangent_x: float
    tangent_y: float
    tangent_slope: float
    tangent_intercept: float
    rmse: float
    y_min: float
    y_max: float
    points_used: int


@dataclass(frozen=True)
class VPResult:
    valid: bool
    vp_x: float
    vp_y: float
    spread: float
    confidence: float
    num_lanes: int
    num_intersections: int
    num_inliers: int
    mean_lane_rmse: float
    reason: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def load_lane_points(annotation_path: Path) -> list[np.ndarray]:
    lanes: list[np.ndarray] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            values = np.fromstring(raw_line, sep=" ", dtype=np.float32)
            if values.size < 4 or values.size % 2 != 0:
                continue
            points = values.reshape(-1, 2)
            lanes.append(points)
    return lanes


def fit_lane_top_segment(
    lane_points: np.ndarray,
    polynomial_degree: int = 2,
    min_points: int = 6,
    top_points: int = 16,
    min_y_span: float = 60.0,
    max_rmse: float = 12.0,
) -> LaneFit | None:
    if lane_points.shape[0] < min_points:
        return None

    order = np.argsort(lane_points[:, 1])
    points = lane_points[order]
    points = points[: min(top_points, points.shape[0])]

    y = points[:, 1]
    x = points[:, 0]

    if float(y.max() - y.min()) < min_y_span:
        return None

    degree = min(polynomial_degree, points.shape[0] - 1)
    if degree < 1:
        return None

    coefficients = np.polyfit(y, x, deg=degree)
    pred_x = np.polyval(coefficients, y)
    rmse = float(np.sqrt(np.mean((pred_x - x) ** 2)))

    if not np.isfinite(rmse) or rmse > max_rmse:
        return None

    tangent_y = float(y.min())
    tangent_x = float(np.polyval(coefficients, tangent_y))
    derivative_coefficients = np.polyder(coefficients)
    tangent_slope = float(np.polyval(derivative_coefficients, tangent_y))
    tangent_intercept = float(tangent_x - tangent_slope * tangent_y)

    return LaneFit(
        polynomial_degree=int(degree),
        coefficients=tuple(float(value) for value in coefficients),
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        tangent_slope=tangent_slope,
        tangent_intercept=tangent_intercept,
        rmse=rmse,
        y_min=float(y.min()),
        y_max=float(y.max()),
        points_used=int(points.shape[0]),
    )


def _pairwise_intersections(
    lane_fits: Sequence[LaneFit],
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    intersections: list[tuple[float, float]] = []
    weights: list[float] = []
    if len(lane_fits) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    ordered_lane_fits = sorted(lane_fits, key=lambda fit: fit.tangent_x)
    candidate_pairs = list(zip(ordered_lane_fits[:-1], ordered_lane_fits[1:]))

    for fit_a, fit_b in candidate_pairs:
        denom = fit_a.tangent_slope - fit_b.tangent_slope
        if abs(denom) < 1e-3:
            continue

        vp_y = (fit_b.tangent_intercept - fit_a.tangent_intercept) / denom
        vp_x = fit_a.tangent_slope * vp_y + fit_a.tangent_intercept

        if not np.isfinite(vp_x) or not np.isfinite(vp_y):
            continue

        if vp_y > image_height * 0.95:
            continue
        if vp_y < -image_height * 1.5:
            continue
        if vp_x < -image_width or vp_x > image_width * 2.0:
            continue

        angle_weight = abs(fit_a.tangent_slope - fit_b.tangent_slope)
        fit_weight = 1.0 / (1.0 + fit_a.rmse + fit_b.rmse)
        center_weight = 1.0 / (1.0 + abs(((fit_a.tangent_x + fit_b.tangent_x) * 0.5) - image_width * 0.5) / image_width)
        intersections.append((float(vp_x), float(vp_y)))
        weights.append(float(angle_weight * fit_weight * center_weight))

    if not intersections:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.asarray(intersections, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def _robust_average(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median[None, :], axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = max(20.0, median_distance + 2.5 * 1.4826 * mad)
    inlier_mask = distances <= threshold

    if not np.any(inlier_mask):
        inlier_mask = np.ones(points.shape[0], dtype=bool)

    inlier_points = points[inlier_mask]
    inlier_weights = weights[inlier_mask]
    if float(inlier_weights.sum()) <= 0.0:
        inlier_weights = np.ones_like(inlier_weights)

    vp = np.average(inlier_points, axis=0, weights=inlier_weights)
    spread = float(
        np.sqrt(
            np.average(
                np.sum((inlier_points - vp[None, :]) ** 2, axis=1),
                weights=inlier_weights,
            )
        )
    )
    return vp, inlier_mask, spread


def estimate_vp_from_lanes(
    lanes: Sequence[np.ndarray],
    image_width: int = 1640,
    image_height: int = 590,
    polynomial_degree: int = 2,
) -> VPResult:
    lane_fits = [
        fit
        for lane in lanes
        if (fit := fit_lane_top_segment(lane, polynomial_degree=polynomial_degree)) is not None
    ]
    if len(lane_fits) < 2:
        return VPResult(
            valid=False,
            vp_x=float("nan"),
            vp_y=float("nan"),
            spread=float("inf"),
            confidence=0.0,
            num_lanes=len(lane_fits),
            num_intersections=0,
            num_inliers=0,
            mean_lane_rmse=float(np.mean([fit.rmse for fit in lane_fits])) if lane_fits else float("inf"),
            reason="not_enough_valid_lanes",
        )

    intersections, weights = _pairwise_intersections(lane_fits, image_width=image_width, image_height=image_height)
    if intersections.shape[0] == 0:
        return VPResult(
            valid=False,
            vp_x=float("nan"),
            vp_y=float("nan"),
            spread=float("inf"),
            confidence=0.0,
            num_lanes=len(lane_fits),
            num_intersections=0,
            num_inliers=0,
            mean_lane_rmse=float(np.mean([fit.rmse for fit in lane_fits])),
            reason="no_valid_intersections",
        )

    vp, inlier_mask, spread = _robust_average(intersections, weights)
    vp_x = float(vp[0])
    vp_y = float(vp[1])
    mean_lane_rmse = float(np.mean([fit.rmse for fit in lane_fits]))

    confidence = (
        min(1.0, len(lane_fits) / 4.0)
        * min(1.0, intersections.shape[0] / 3.0)
        * float(np.exp(-spread / 45.0))
        * float(np.exp(-mean_lane_rmse / 8.0))
    )

    valid = True
    reason = "ok"

    if vp_x < -image_width * 0.5 or vp_x > image_width * 1.5:
        valid = False
        reason = "vp_x_out_of_bounds"
    elif vp_y < -image_height * 1.5 or vp_y > image_height * 0.95:
        valid = False
        reason = "vp_y_out_of_bounds"
    elif spread > 55.0:
        valid = False
        reason = "intersection_spread_too_large"
    elif confidence < 0.15:
        valid = False
        reason = "low_confidence"

    return VPResult(
        valid=valid,
        vp_x=vp_x,
        vp_y=vp_y,
        spread=spread,
        confidence=confidence,
        num_lanes=len(lane_fits),
        num_intersections=int(intersections.shape[0]),
        num_inliers=int(inlier_mask.sum()),
        mean_lane_rmse=mean_lane_rmse,
        reason=reason,
    )
