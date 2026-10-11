"""Step 3b: run the two-view pipeline on two real TUM frames and compare to ground truth.

Usage:
    python notebooks/two_view_check.py data/rgbd_dataset_freiburg1_xyz --i 100 --gap 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam.dataset import load_tum_sequence, tum_fr1_camera
from slam.features import detect_orb, match_descriptors
from slam.geometry import estimate_essential, recover_pose, triangulate_points
from slam.viz import draw_matches, plot_two_view_scene


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--i", type=int, default=100, help="index of the first frame")
    parser.add_argument("--gap", type=int, default=8, help="frames between the two images")
    args = parser.parse_args()

    camera = tum_fr1_camera()
    frames = load_tum_sequence(args.root)
    f1, f2 = frames[args.i], frames[args.i + args.gap]
    img1 = cv2.imread(str(f1.image_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(f2.image_path), cv2.IMREAD_GRAYSCALE)

    kps1, des1 = detect_orb(img1)
    kps2, des2 = detect_orb(img2)
    matches = match_descriptors(des1, des2)
    pix1 = np.array([kps1[m.queryIdx].pt for m in matches])
    pix2 = np.array([kps2[m.trainIdx].pt for m in matches])
    print(f"keypoints: {len(kps1)}, {len(kps2)}; ratio-test matches: {len(matches)}")

    E, inliers = estimate_essential(pix1, pix2, camera)
    if E is None:
        raise SystemExit("estimate_essential failed (too few matches?)")
    print(f"RANSAC inliers: {inliers.sum()} / {len(inliers)} ({inliers.mean():.0%})")

    T_cam2_cam1 = recover_pose(E, pix1[inliers], pix2[inliers], camera)

    # Ground truth: T_cam2_cam1 = inv(T_world_cam2) @ T_world_cam1
    T_gt = np.linalg.inv(f2.T_world_cam) @ f1.T_world_cam
    R_err = rotation_angle_deg(T_cam2_cam1[:3, :3].T @ T_gt[:3, :3])
    t_err = angle_between_deg(T_cam2_cam1[:3, 3], T_gt[:3, 3])
    print(f"GT rotation angle: {rotation_angle_deg(T_gt[:3, :3]):.2f} deg, "
          f"GT baseline: {np.linalg.norm(T_gt[:3, 3]):.3f} m")
    print(f"rotation error: {R_err:.2f} deg, translation direction error: {t_err:.2f} deg")

    # Triangulate inliers in the camera-1 frame (P1 = K [I | 0], P2 = K [R | t]).
    P1 = camera.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = camera.K @ T_cam2_cam1[:3]
    pts1 = triangulate_points(P1, P2, pix1[inliers], pix2[inliers])
    pts2 = pts1 @ T_cam2_cam1[:3, :3].T + T_cam2_cam1[:3, 3]
    in_front = (pts1[:, 2] > 0) & (pts2[:, 2] > 0)
    print(f"points in front of both cameras: {in_front.sum()} / {len(pts1)}")

    cv2.imshow("matches (green=inlier, red=outlier)", draw_matches(img1, img2, pix1, pix2, inliers))
    cv2.waitKey(1)
    good = pts1[in_front]
    z_hi = np.percentile(good[:, 2], 95)  # clip far outliers so the plot stays readable
    plot_two_view_scene(good[good[:, 2] < z_hi], T_cam2_cam1, T_gt)
    plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
