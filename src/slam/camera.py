"""Step 1: pinhole camera model.

Conventions (keep these everywhere in the project):
- Right-handed camera frame: x right, y down, z forward (OpenCV convention).
- Pixel coordinates (u, v): u to the right, v downward, origin at the top-left corner.
- Units: meters for 3D points, pixels for image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """Ideal pinhole camera without lens distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project 3D points given in the camera frame to pixel coordinates.

        Args:
            points_cam: (N, 3) array of points in the camera frame.

        Returns:
            (N, 2) array of pixel coordinates (u, v). Points with z <= 0 (behind or on
            the camera plane) must come back as NaN, not as a wrong-looking pixel.

        Hint: u = fx * x / z + cx, v = fy * y / z + cy. Do it vectorized, no Python loop.
        """
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        valid_mask = z > 0
        u = self.fx * x[valid_mask] / z[valid_mask] + self.cx
        v = self.fy * y[valid_mask] / z[valid_mask] + self.cy
        out = np.full((points_cam.shape[0], 2), np.nan)
        out[valid_mask] = np.concatenate((u[:, None], v[:, None]), axis=1)
        return out

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Back-project pixels with known depth to 3D points in the camera frame.

        Args:
            pixels: (N, 2) array of pixel coordinates (u, v).
            depth: (N,) array of z values (meters along the optical axis).

        Returns:
            (N, 3) array of points in the camera frame.

        Hint: invert the projection equations, then scale by the depth.
        """
        u, v = pixels[:, 0], pixels[:, 1]
        x = (u - self.cx) / self.fx * depth
        y = (v - self.cy) / self.fy * depth
        z = depth
        return np.stack((x, y, z), axis=1)
