import cv2
import numpy as np

from slam.camera import PinholeCamera
from slam.geometry import estimate_essential, recover_pose, triangulate_points

CAM = PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def make_scene(n: int = 200, seed: int = 0):
    """Random points in front of camera 1 and their pixels in both views."""
    rng = np.random.default_rng(seed)
    points_cam1 = rng.uniform([-2, -1.5, 4], [2, 1.5, 9], size=(n, 3))
    rvec = np.array([0.02, 0.1, -0.03])
    R_cam2_cam1, _ = cv2.Rodrigues(rvec)
    t_cam2_cam1 = np.array([-0.6, 0.05, 0.1])
    points_cam2 = points_cam1 @ R_cam2_cam1.T + t_cam2_cam1
    pixels1 = CAM.project(points_cam1)
    pixels2 = CAM.project(points_cam2)
    return points_cam1, pixels1, pixels2, R_cam2_cam1, t_cam2_cam1


def test_triangulate_recovers_points():
    points_cam1, pixels1, pixels2, R, t = make_scene()
    P1 = CAM.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = CAM.K @ np.hstack([R, t[:, None]])
    recovered = triangulate_points(P1, P2, pixels1, pixels2)
    np.testing.assert_allclose(recovered, points_cam1, atol=1e-6)


def test_estimate_essential_satisfies_epipolar_constraint():
    _, pixels1, pixels2, _, _ = make_scene()
    E, mask = estimate_essential(pixels1, pixels2, CAM)
    assert mask.shape == (len(pixels1),)
    assert mask.mean() > 0.95
    x1 = np.hstack([pixels1, np.ones((len(pixels1), 1))]) @ np.linalg.inv(CAM.K).T
    x2 = np.hstack([pixels2, np.ones((len(pixels2), 1))]) @ np.linalg.inv(CAM.K).T
    residual = np.abs(np.einsum("ni,ij,nj->n", x2, E, x1))
    assert np.median(residual) < 1e-3


def test_estimate_essential_rejects_outliers():
    _, pixels1, pixels2, _, _ = make_scene(n=300)
    rng = np.random.default_rng(1)
    bad = rng.choice(len(pixels2), size=60, replace=False)
    pixels2 = pixels2.copy()
    pixels2[bad] = rng.uniform([0, 0], [640, 480], size=(60, 2))
    _, mask = estimate_essential(pixels1, pixels2, CAM)
    assert mask[bad].mean() < 0.2  # most injected outliers are rejected
    good = np.setdiff1d(np.arange(len(pixels1)), bad)
    assert mask[good].mean() > 0.9  # most true matches are kept


def test_recover_pose_matches_ground_truth_up_to_scale():
    _, pixels1, pixels2, R_true, t_true = make_scene()
    E, _ = estimate_essential(pixels1, pixels2, CAM)
    T_cam2_cam1 = recover_pose(E, pixels1, pixels2, CAM)
    assert T_cam2_cam1.shape == (4, 4)
    R, t = T_cam2_cam1[:3, :3], T_cam2_cam1[:3, 3]
    np.testing.assert_allclose(R, R_true, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(t), 1.0, atol=1e-6)
    np.testing.assert_allclose(t, t_true / np.linalg.norm(t_true), atol=1e-3)
