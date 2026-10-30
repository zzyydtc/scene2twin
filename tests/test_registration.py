import numpy as np
import pytest
from scipy.spatial import cKDTree

from eval.colmap_io import quat_wxyz_to_matrix
from registration.pipeline import (
    compose, icp_sim3, inverse_similarity, mutual_matches, normalise, ransac_sim3, similarity,
)
from registration.sim3 import Sim3, umeyama

RNG = np.random.default_rng(1)


def random_sim3(scale=1.7) -> Sim3:
    return Sim3(scale, quat_wxyz_to_matrix(RNG.normal(size=4)), RNG.normal(size=3))


def test_compose_matches_sequential_application():
    a, b = random_sim3(0.5), random_sim3(2.0)
    x = RNG.normal(size=(10, 3))
    np.testing.assert_allclose(compose(a, b).apply(x), a.apply(b.apply(x)), atol=1e-12)


def test_similarity_and_inverse_roundtrip():
    centre, scale = np.array([1.0, 2.0, 3.0]), 4.2
    x = RNG.normal(size=(10, 3))
    back = inverse_similarity(scale, centre).apply(similarity(scale, centre).apply(x))
    np.testing.assert_allclose(back, x, atol=1e-12)


def test_normalise_gives_unit_rms():
    pts, centre, rms = normalise(RNG.normal(size=(500, 3)) * 3 + 5)
    assert np.sqrt((pts ** 2).sum(axis=1).mean()) == pytest.approx(1.0)
    np.testing.assert_allclose(pts.mean(axis=0), 0, atol=1e-12)


def test_ransac_recovers_transform_with_outliers():
    src = RNG.normal(size=(300, 3))
    truth = random_sim3()
    dst = truth.apply(src) + RNG.normal(scale=0.002, size=src.shape)
    bad = RNG.choice(300, size=180, replace=False)  # 60% wrong correspondences
    dst[bad] = RNG.normal(size=(180, 3)) * 3
    model, inliers = ransac_sim3(src, dst, tau=0.02, iters=5000, rng=np.random.default_rng(0),
                                 scale_range=(0.1, 10))
    assert model.scale == pytest.approx(truth.scale, rel=0.01)
    assert inliers.sum() >= 100


def test_icp_converges_from_perturbed_init_and_recovers_scale():
    dst = RNG.uniform(size=(2000, 3)) * [3, 2, 1]
    truth = random_sim3(1.0)
    src = RNG.permutation(dst)  # same shape, different order
    src = np.linalg.inv(truth.scale * truth.rotation) @ (src - truth.translation).T
    src = src.T
    init = Sim3(truth.scale * 1.03, truth.rotation, truth.translation + 0.01)
    est = icp_sim3(src, dst, init, max_dist=0.2, iters=100, trim=1.0)
    resid = cKDTree(dst).query(est.apply(src))[0]  # src is a permutation of dst's points
    assert np.median(resid) < 0.02
    assert est.scale == pytest.approx(truth.scale, rel=0.02)


def test_mutual_matches_identical_features():
    feat = RNG.normal(size=(50, 33))
    pairs = mutual_matches(feat, feat)
    assert len(pairs) == 50
    np.testing.assert_array_equal(pairs[:, 0], pairs[:, 1])


def test_umeyama_requires_three_points():
    with pytest.raises(ValueError):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))


def test_icp_scale_bounds_prevent_collapse():
    # Without a bound, scaled ICP on a one-sided objective can shrink the source.
    dst = RNG.uniform(size=(3000, 3)) * [4, 3, 2]
    src = dst[RNG.choice(3000, size=300, replace=False)] + RNG.normal(scale=0.05, size=(300, 3))
    init = Sim3(1.0, np.eye(3), np.zeros(3))
    est = icp_sim3(src, dst, init, max_dist=3.0, iters=100, trim=0.5, scale_bounds=(0.85, 1.15))
    assert 0.85 - 1e-9 <= est.scale <= 1.15 + 1e-9
