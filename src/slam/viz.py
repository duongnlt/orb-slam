"""Plotting helpers (visualization only, no SLAM logic)."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    inlier_mask: np.ndarray,
    max_draw: int = 200,
) -> np.ndarray:
    """Side-by-side image with match lines: green = inlier, red = outlier.

    Args:
        image1, image2: grayscale or BGR uint8 images of the same size.
        pixels1, pixels2: (N, 2) matching pixel coordinates.
        inlier_mask: (N,) bool.
        max_draw: random subsample size, so the picture stays readable.

    Returns:
        BGR uint8 image of shape (H, 2W, 3).
    """
    to_bgr = lambda im: cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im  # noqa: E731
    canvas = np.hstack([to_bgr(image1), to_bgr(image2)])
    w = image1.shape[1]
    idx = np.arange(len(pixels1))
    if len(idx) > max_draw:
        idx = np.random.default_rng(0).choice(idx, max_draw, replace=False)
    for i in idx:
        color = (0, 200, 0) if inlier_mask[i] else (0, 0, 220)
        p1 = tuple(np.round(pixels1[i]).astype(int))
        p2 = tuple(np.round(pixels2[i] + [w, 0]).astype(int))
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1)
        cv2.circle(canvas, p2, 2, color, -1)
    return canvas


def plot_two_view_scene(
    points_cam1: np.ndarray,
    T_cam2_cam1: np.ndarray,
    T_cam2_cam1_gt: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Top-down (x-z plane) view of triangulated points and both camera centers.

    Everything is drawn in the camera-1 frame. Camera 1 is at the origin; camera 2's
    center is -R^T t. The ground-truth translation is shown if given (unit length, since
    monocular translation has no scale).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points_cam1[:, 0], points_cam1[:, 2], s=3, c="gray", label="points")

    def center2(T: np.ndarray) -> np.ndarray:
        return -T[:3, :3].T @ T[:3, 3]

    ax.scatter([0], [0], c="blue", marker="^", s=80, label="cam1")
    c2 = center2(T_cam2_cam1)
    ax.scatter([c2[0]], [c2[2]], c="green", marker="^", s=80, label="cam2 (est)")
    if T_cam2_cam1_gt is not None:
        t_gt = T_cam2_cam1_gt[:3, 3] / np.linalg.norm(T_cam2_cam1_gt[:3, 3])
        c2_gt = -T_cam2_cam1_gt[:3, :3].T @ t_gt
        ax.scatter([c2_gt[0]], [c2_gt[2]], c="red", marker="^", s=80, label="cam2 (gt dir)")
    ax.set_xlabel("x [right]")
    ax.set_ylabel("z [forward]")
    ax.set_aspect("equal")
    ax.legend()
    return ax
