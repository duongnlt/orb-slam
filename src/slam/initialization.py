from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam.camera import PinholeCamera
from slam.geometry import estimate_essential, recover_pose, triangulate_points


@dataclass(frozen=True)
class InitConfig:
    min_inliers: int = 100
    ransac_threshold_px: float = 1.0
    max_reproj_px: float = 2.0
    min_parallax_deg: float = 1.0


@dataclass(frozen=True)
class Initialization:
    T_cam2_cam1: np.ndarray
    points_cam1: np.ndarray
    inlier_mask: np.ndarray
    parallax_deg: float


def parallax_angles(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    T_cam2_cam1: np.ndarray,
    camera: PinholeCamera,
) -> np.ndarray:
    """Angle (radians) at each 3D point between the rays to the two camera centers.

    Args:
        pixels1, pixels2: (N, 2) matched pixels in view 1 and view 2.
        T_cam2_cam1: (4, 4) transform taking cam1 coordinates to cam2 coordinates.
    """
    n = pixels1.shape[0]
    rays1 = camera.unproject(pixels1, np.ones(n))
    rays2 = camera.unproject(pixels2, np.ones(n))
    rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)
    rays2 /= np.linalg.norm(rays2, axis=1, keepdims=True)

    # Row-vector form of R^T @ b2: bring the cam2 rays into the cam1 frame.
    rays2_in_1 = rays2 @ T_cam2_cam1[:3, :3]

    cross_norm = np.linalg.norm(np.cross(rays1, rays2_in_1), axis=1)
    dot = np.sum(rays1 * rays2_in_1, axis=1)
    return np.arctan2(cross_norm, dot)


def reprojection_errors(
    points_cam1: np.ndarray,
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    T_cam2_cam1: np.ndarray,
    camera: PinholeCamera,
) -> np.ndarray:
    """Per-point reprojection error in pixels (max over the two views).

    Points behind either camera come back as NaN, so `err < threshold` rejects them.
    """
    points_cam2 = points_cam1 @ T_cam2_cam1[:3, :3].T + T_cam2_cam1[:3, 3]

    err1 = np.linalg.norm(camera.project(points_cam1) - pixels1, axis=1)
    err2 = np.linalg.norm(camera.project(points_cam2) - pixels2, axis=1)
    return np.maximum(err1, err2)


def initialize(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    camera: PinholeCamera,
    config: InitConfig,
) -> Initialization | None:
    """Try to initialize a monocular map from one pair of matched pixels.

    Returns None if the pair is rejected (too few inliers, too little parallax,
    or poor triangulation). On success the scale is fixed so the median depth in
    camera 1 is 1, and `points_cam1` holds one row per True entry of `inlier_mask`.
    """
    E, ransac_mask = estimate_essential(
        pixels1, pixels2, camera, threshold_px=config.ransac_threshold_px
    )
    if E is None or ransac_mask is None:
        return None
    E = E[:3]  # OpenCV may stack several candidate solutions; use the first
    if ransac_mask.sum() < config.min_inliers:
        return None

    idx = np.flatnonzero(ransac_mask)
    p1, p2 = pixels1[idx], pixels2[idx]

    T_cam2_cam1 = recover_pose(E, p1, p2, camera)
    R, t = T_cam2_cam1[:3, :3], T_cam2_cam1[:3, 3]

    P1 = camera.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = camera.K @ T_cam2_cam1[:3, :]
    points_cam1 = triangulate_points(P1, P2, p1, p2)
    points_cam2 = points_cam1 @ R.T + t

    good = np.isfinite(points_cam1).all(axis=1)
    good &= (points_cam1[:, 2] > 0) & (points_cam2[:, 2] > 0)
    # NaN errors (points behind a camera) compare False, so they are rejected too.
    errors = reprojection_errors(points_cam1, p1, p2, T_cam2_cam1, camera)
    good &= errors < config.max_reproj_px
    if good.sum() < config.min_inliers:
        return None

    angles = parallax_angles(p1[good], p2[good], T_cam2_cam1, camera)
    parallax_deg = float(np.degrees(np.median(angles)))
    if parallax_deg < config.min_parallax_deg:
        return None

    # Monocular scale is arbitrary: pick median depth in camera 1 = 1.
    scale = np.median(points_cam1[good, 2])
    T_scaled = T_cam2_cam1.copy()
    T_scaled[:3, 3] = t / scale

    inlier_mask = np.zeros(pixels1.shape[0], dtype=bool)
    inlier_mask[idx[good]] = True
    return Initialization(
        T_cam2_cam1=T_scaled,
        points_cam1=points_cam1[good] / scale,
        inlier_mask=inlier_mask,
        parallax_deg=parallax_deg,
    )
