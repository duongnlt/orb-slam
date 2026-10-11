"""Step 2: ORB feature detection and matching.

Matching strategy used by ORB-SLAM-style systems: binary descriptors compared with
Hamming distance, then a ratio test to throw away ambiguous matches. RANSAC-based
geometric verification comes later (step 3), once we have a motion model.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_orb(image: np.ndarray, n_features: int = 1000) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect ORB keypoints and compute their descriptors.

    Args:
        image: grayscale uint8 image of shape (H, W).
        n_features: maximum number of features to keep.

    Returns:
        (keypoints, descriptors) where descriptors has shape (len(keypoints), 32), uint8.
        If nothing is found, return ([], empty (0, 32) uint8 array).
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    kps = orb.detect(image)
    kps, des = orb.compute(image, kps)
    if des is None or des.shape[0] == 0:
        return [], np.empty((0, 32), dtype=np.uint8)
    return kps, des


def match_descriptors(
    desc_a: np.ndarray, desc_b: np.ndarray, ratio: float = 0.75
) -> list[cv2.DMatch]:
    """Match binary descriptors from image A to image B with Lowe's ratio test.

    Args:
        desc_a: (Na, 32) uint8 descriptors.
        desc_b: (Nb, 32) uint8 descriptors.
        ratio: keep a match only if best_distance < ratio * second_best_distance.

    Returns:
        List of cv2.DMatch with queryIdx indexing desc_a and trainIdx indexing desc_b.
    """
    # raise NotImplementedError("TODO: implement match_descriptors()")
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc_a, desc_b, k=2)
    ratio_matches = []
    for match in matches:
        if len(match) < 2:
            continue
        m, n = match
        if m.distance < n.distance * ratio:
            ratio_matches.append(m)
    return ratio_matches
