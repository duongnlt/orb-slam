"""Step 4a: TUM RGB-D dataset loading (pure IO, no SLAM logic).

TUM file formats (lines starting with '#' are comments):
- rgb.txt:          timestamp  rgb/<timestamp>.png
- groundtruth.txt:  timestamp  tx ty tz qx qy qz qw   (pose of the camera in the world,
                    i.e. T_world_cam; the quaternion order is x, y, z, w)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slam.camera import PinholeCamera


@dataclass(frozen=True)
class Frame:
    """One image with its ground-truth camera pose."""

    timestamp: float
    image_path: Path
    T_world_cam: np.ndarray  # (4, 4)


def tum_fr1_camera() -> PinholeCamera:
    """Intrinsics of the TUM freiburg1 sequences.

    fx = 517.3, fy = 516.5, cx = 318.6, cy = 255.3, 640 x 480. The real images also have
    lens distortion (d = 0.2624, -0.9531, -0.0054, 0.0026, 1.1633); decide whether to
    undistort them when loading or to ignore it at first.
    """
    # raise NotImplementedError("TODO: implement tum_fr1_camera()")
    return PinholeCamera(fx = 517.3, fy = 516.5, cx = 318.6, cy = 255.3, width=640, height=480)


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix.

    Hint: R = I + 2*qw*[v]x + 2*[v]x^2 with v = (qx, qy, qz), or expand the usual
    formula. Normalize the input first; file values are rounded.
    """
    q = np.array([qx, qy, qz, qw])
    qx, qy, qz, qw = q / np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    K = np.array([
        [0, -qz, qy],
        [qz, 0, -qx],
        [-qy, qx, 0]
    ])
    R = np.eye(3) + 2*qw*K + 2 * K @ K
    return R


def load_tum_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read groundtruth.txt.

    Returns:
        (timestamps, poses): timestamps has shape (M,), poses has shape (M, 4, 4) and
        holds T_world_cam for each timestamp.
    """
    gt = np.loadtxt(path)
    timestamps = gt[:, 0]
    t = gt[:, 1:4]
    Q = gt[:, 4:]
    R = np.array([quaternion_to_rotation(*q) for q in Q])
    poses = np.tile(np.eye(4), (Q.shape[0], 1, 1))
    poses[:, :3, :3] = R
    poses[:, :3, 3] = t 
    return timestamps, poses
    # raise NotImplementedError("TODO: implement load_tum_poses()")


def load_tum_images(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read rgb.txt.

    Returns:
        (timestamps, relative_paths): timestamps has shape (M,); relative_paths are the
        strings from the file (relative to the sequence folder).
    """
    # raise NotImplementedError("TODO: implement load_tum_images()")
    rgb = np.loadtxt(path, dtype=str)
    return rgb[:, 0].astype(float), rgb[:, 1].tolist()
    


def associate(
    timestamps_a: np.ndarray, timestamps_b: np.ndarray, max_diff: float = 0.02
) -> list[tuple[int, int]]:
    """Match each timestamp in A to the nearest timestamp in B.

    Returns:
        List of (index_in_a, index_in_b) for pairs closer than max_diff seconds. An entry
        of A with no close partner is dropped. Each index in A appears at most once.

    Hint: both arrays are sorted, so np.searchsorted gives the candidates without a loop
    over every pair. Check the neighbour on both sides of the insertion point.
    """
    # raise NotImplementedError("TODO: implement associate()")
    i = np.searchsorted(timestamps_b, timestamps_a)
    left = np.clip(i - 1, 0, len(timestamps_b)-1)
    right = np.clip(i, 0, len(timestamps_b)-1)
    
    d_left = np.abs(timestamps_a - timestamps_b[left])
    d_right = np.abs(timestamps_a - timestamps_b[right])
    
    j = np.where(d_right < d_left, right, left)
    d = np.minimum(d_left, d_right)
    keep = d <= max_diff
    
    return list(zip(np.flatnonzero(keep).tolist(), j[keep].tolist(), strict=True))


def load_tum_sequence(root: Path, max_diff: float = 0.02) -> list[Frame]:
    """Load a TUM sequence folder into a list of Frames sorted by time.

    Images without a ground-truth pose within max_diff seconds are skipped.
    """
    # raise NotImplementedError("TODO: implement load_tum_sequence()")
    img_ts, rgb_path = load_tum_images(root / "rgb.txt")
    gt_ts, poses = load_tum_poses(root / "groundtruth.txt")
    pairs = associate(img_ts, gt_ts, max_diff)
    return [Frame(img_ts[a], root / rgb_path[a], poses[b]) for a,b in pairs]
    
