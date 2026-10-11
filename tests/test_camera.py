import numpy as np

from slam.camera import PinholeCamera

CAM = PinholeCamera(fx=500.0, fy=450.0, cx=320.0, cy=240.0, width=640, height=480)


def test_intrinsic_matrix():
    expected = np.array([[500.0, 0.0, 320.0], [0.0, 450.0, 240.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(CAM.K, expected)


def test_point_on_optical_axis_projects_to_principal_point():
    uv = CAM.project(np.array([[0.0, 0.0, 2.0]]))
    np.testing.assert_allclose(uv, [[320.0, 240.0]])


def test_project_known_point():
    # x/z = 0.5 -> u = 500 * 0.5 + 320; y/z = -0.25 -> v = 450 * -0.25 + 240
    uv = CAM.project(np.array([[1.0, -0.5, 2.0]]))
    np.testing.assert_allclose(uv, [[570.0, 127.5]])


def test_points_behind_camera_give_nan():
    uv = CAM.project(np.array([[0.1, 0.1, -1.0], [0.1, 0.1, 0.0], [0.1, 0.1, 1.0]]))
    assert np.isnan(uv[0]).all()
    assert np.isnan(uv[1]).all()
    assert np.isfinite(uv[2]).all()


def test_project_unproject_roundtrip():
    rng = np.random.default_rng(0)
    points = rng.uniform([-2, -2, 1], [2, 2, 5], size=(100, 3))
    uv = CAM.project(points)
    recovered = CAM.unproject(uv, points[:, 2])
    np.testing.assert_allclose(recovered, points, atol=1e-9)
