"""Step 3: two-view geometry.

Conventions:
- T_cam2_cam1 maps points from camera 1 coordinates to camera 2 coordinates:
  p_cam2 = R @ p_cam1 + t.
- With monocular images the translation is only known up to scale, so recovered
  poses use ||t|| = 1.
"""

from __future__ import annotations

import numpy as np
import cv2

from slam.camera import PinholeCamera


def triangulate_points(
    P1: np.ndarray, P2: np.ndarray, pixels1: np.ndarray, pixels2: np.ndarray
) -> np.ndarray:
    """Triangulate 3D points from two views with the linear (DLT) method.

    Args:
        P1, P2: (3, 4) projection matrices K @ [R | t] of the two views.
        pixels1, pixels2: (N, 2) matching pixel coordinates in view 1 and view 2.

    Returns:
        (N, 3) points expressed in the frame that P1 and P2 are defined in.

    Hint: each view gives two linear equations on the homogeneous point X, e.g.
    u * P[2] - P[0] and v * P[2] - P[1] (rows of P). Stack four rows per point,
    solve A X = 0 with an SVD, then dehomogenize.
    """
    u1, v1 = pixels1[:, 0], pixels1[:, 1]
    u2, v2 = pixels2[:, 0], pixels2[:, 1]
    A = np.stack(
        [
            u1[:, None] * P1[2, :] - P1[0, :],
            v1[:, None] * P1[2, :] - P1[1, :],
            u2[:, None] * P2[2, :] - P2[0, :],
            v2[:, None] * P2[2, :] - P2[1, :],
        ],
        axis=1
    )
    U, S, Vt = np.linalg.svd(A)
    X_homo = Vt[:, -1, :]
    X = X_homo[:, :3] / X_homo[:, 3:4]
    return X
    


def estimate_essential(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    camera: PinholeCamera,
    threshold_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the essential matrix from matches, robust to outliers (RANSAC).

    Args:
        pixels1, pixels2: (N, 2) matching pixel coordinates in image 1 and image 2.
        camera: intrinsics shared by both images.
        threshold_px: inlier threshold in pixels.

    Returns:
        (E, inlier_mask) where E is 3x3 and inlier_mask is a (N,) bool array.

    Hint: the epipolar constraint x2^T E x1 = 0 holds in normalized coordinates
    (inv(K) applied to the pixels). Start with cv2.findEssentialMat, then ask
    yourself what RANSAC is doing inside it.
    """
    # raise NotImplementedError("TODO: implement estimate_essential()")
    camera_matrix = camera.K
    try:
        E, inlier_mask  = cv2.findEssentialMat(pixels1, pixels2, cameraMatrix=camera_matrix, method=cv2.RANSAC, threshold=threshold_px)
    except:
        return None, None
    
    return E, inlier_mask.ravel().astype(np.bool)
    


def recover_pose(
    E: np.ndarray, pixels1: np.ndarray, pixels2: np.ndarray, camera: PinholeCamera
) -> np.ndarray:
    """Decompose E into the relative pose T_cam2_cam1 (4x4, with ||t|| = 1).

    Hint: E has four (R, t) candidates (two rotations times the sign of t). Only one
    puts the triangulated points in front of both cameras (z > 0). That check is the
    cheirality test.
    """
    # raise NotImplementedError("TODO: implement recover_pose()")
    U, S, Vt = np.linalg.svd(E)
    
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    t = t / np.linalg.norm(t)
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    best_candidate = 0
    best_cnt = 0
    for id, (R, t) in enumerate(candidates):
        P1 = camera.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = camera.K @ np.hstack((R, t.reshape(3,1)))
        
        X = triangulate_points(P1, P2, pixels1, pixels2)
        
        Z1 = X[:, 2]
        X2 = X @ R.T + t.ravel()
        Z2 = X2[:, 2]
        
        cnt = np.count_nonzero((Z1 > 0) & (Z2 > 0))
        if cnt > best_cnt:
            best_cnt = cnt
            best_candidate = id
    R, t = candidates[best_candidate]
    T_cam2_cam1 = np.eye(4)
    T_cam2_cam1[:3, :3] = R
    T_cam2_cam1[:3, -1] = t
    return T_cam2_cam1
        
        
        
    
