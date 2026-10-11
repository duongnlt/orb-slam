from pathlib import Path

import cv2
import numpy as np

from slam.dataset import (
    associate,
    load_tum_images,
    load_tum_poses,
    load_tum_sequence,
    quaternion_to_rotation,
    tum_fr1_camera,
)

S = np.sqrt(0.5)


def make_fake_sequence(root: Path) -> None:
    """Three images; the middle one has no ground truth within 0.02 s."""
    (root / "rgb").mkdir()
    for ts in ("1.000000", "1.100000", "1.200000"):
        cv2.imwrite(str(root / "rgb" / f"{ts}.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    (root / "rgb.txt").write_text(
        "# color images\n# timestamp filename\n"
        "1.000000 rgb/1.000000.png\n"
        "1.100000 rgb/1.100000.png\n"
        "1.200000 rgb/1.200000.png\n"
    )
    (root / "groundtruth.txt").write_text(
        "# ground truth\n# timestamp tx ty tz qx qy qz qw\n"
        f"1.005000 1.0 2.0 3.0 0.0 0.0 {S} {S}\n"  # 90 degrees about z
        "1.190000 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
    )


def test_tum_fr1_camera_intrinsics():
    cam = tum_fr1_camera()
    np.testing.assert_allclose(
        cam.K, [[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]]
    )
    assert (cam.width, cam.height) == (640, 480)


def test_quaternion_identity_and_90_degrees_about_z():
    np.testing.assert_allclose(quaternion_to_rotation(0, 0, 0, 1), np.eye(3), atol=1e-12)
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(quaternion_to_rotation(0, 0, S, S), expected, atol=1e-12)


def test_quaternion_gives_proper_rotation_even_if_not_normalized():
    R = quaternion_to_rotation(0.1, 0.2, 0.3, 0.9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_load_tum_poses_skips_comments_and_builds_transforms(tmp_path):
    make_fake_sequence(tmp_path)
    timestamps, poses = load_tum_poses(tmp_path / "groundtruth.txt")
    np.testing.assert_allclose(timestamps, [1.005, 1.19])
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses[0, :3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(poses[0, :3, :3][0], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(poses[0, 3], [0.0, 0.0, 0.0, 1.0])


def test_load_tum_images(tmp_path):
    make_fake_sequence(tmp_path)
    timestamps, names = load_tum_images(tmp_path / "rgb.txt")
    np.testing.assert_allclose(timestamps, [1.0, 1.1, 1.2])
    assert names[0] == "rgb/1.000000.png"


def test_associate_picks_nearest_and_drops_far_entries():
    a = np.array([1.0, 1.1, 1.2])
    b = np.array([1.005, 1.19])
    assert associate(a, b, max_diff=0.02) == [(0, 0), (2, 1)]
    assert associate(a, b, max_diff=0.001) == []


def test_load_tum_sequence(tmp_path):
    make_fake_sequence(tmp_path)
    frames = load_tum_sequence(tmp_path)
    assert [f.timestamp for f in frames] == [1.0, 1.2]
    assert frames[0].image_path == tmp_path / "rgb" / "1.000000.png"
    assert frames[0].image_path.exists()
    np.testing.assert_allclose(frames[0].T_world_cam[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frames[1].T_world_cam, np.eye(4), atol=1e-12)
