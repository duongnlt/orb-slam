import cv2
import numpy as np

from slam.features import detect_orb, match_descriptors


def make_textured_image(seed: int = 0, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Synthetic grayscale image with plenty of corners and no repetitive pattern."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = rng.integers(0, 60, size=(h, w), dtype=np.uint8)
    for _ in range(150):
        x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        x1, y1 = x0 + int(rng.integers(8, 40)), y0 + int(rng.integers(8, 40))
        color = int(rng.integers(0, 256))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=-1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_detect_orb_output_shapes():
    img = make_textured_image()
    keypoints, descriptors = detect_orb(img, n_features=500)
    assert 0 < len(keypoints) <= 500
    assert descriptors.shape == (len(keypoints), 32)
    assert descriptors.dtype == np.uint8


def test_detect_orb_on_blank_image_returns_nothing():
    keypoints, descriptors = detect_orb(np.zeros((240, 320), dtype=np.uint8))
    assert len(keypoints) == 0
    assert descriptors.shape == (0, 32)


def test_matches_recover_known_translation():
    dx, dy = 12.0, 7.0
    img_a = make_textured_image()
    shift = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
    img_b = cv2.warpAffine(img_a, shift, (img_a.shape[1], img_a.shape[0]))

    kp_a, desc_a = detect_orb(img_a, n_features=1000)
    kp_b, desc_b = detect_orb(img_b, n_features=1000)
    matches = match_descriptors(desc_a, desc_b, ratio=0.75)

    assert len(matches) >= 50
    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches])
    displacement = pts_b - pts_a
    # Most matches must agree with the true motion (a few outliers are fine here;
    # RANSAC in step 3 handles those).
    err = np.linalg.norm(displacement - np.array([dx, dy]), axis=1)
    assert np.mean(err < 2.0) > 0.8


def test_ratio_test_is_stricter_with_smaller_ratio():
    img = make_textured_image(seed=1)
    _, desc = detect_orb(img, n_features=800)
    loose = match_descriptors(desc, desc[::-1].copy(), ratio=0.9)
    strict = match_descriptors(desc, desc[::-1].copy(), ratio=0.5)
    assert len(strict) <= len(loose)
