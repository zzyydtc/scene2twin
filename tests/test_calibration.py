import numpy as np
import pytest

from eval.calibration import ause, reliability, sparsification, track_geometry


def test_two_ray_covariance_matches_the_textbook_formula():
    """Two cameras 2 m apart looking at a point 4 m away along z: depth is the weak direction."""
    f = 1000.0
    centres = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    x = np.array([0.0, 0.0, 4.0])
    cov, angle, mean_range = track_geometry(x, centres, np.array([f, f]))
    # Each ray has perpendicular noise r / f. Two rays at +-theta to the z axis intersect with
    # depth noise (r / f) / (sqrt(2) * sin(theta)).
    r = np.sqrt(17.0)
    sin_theta = 1.0 / r
    sigma_z = np.sqrt(cov[2, 2])
    assert sigma_z == pytest.approx((r / f) / (np.sqrt(2.0) * sin_theta), rel=1e-3)
    assert np.sqrt(cov[1, 1]) < sigma_z  # lateral precision is better than depth precision
    assert angle == pytest.approx(2 * np.degrees(np.arctan(0.25)), rel=1e-6)
    assert mean_range == pytest.approx(np.sqrt(17.0))


def test_more_rays_improve_precision():
    f = 1000.0
    x = np.array([0.0, 0.0, 4.0])
    two = track_geometry(x, np.array([[-1.0, 0, 0], [1.0, 0, 0]]), np.array([f, f]))[0]
    four = track_geometry(x, np.array([[-1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]]), np.full(4, f))[0]
    assert np.trace(four) < np.trace(two)


def test_ause_is_zero_for_a_perfect_ranking_and_positive_for_random():
    err = np.abs(np.random.default_rng(0).normal(size=500))
    assert ause(err, err) == pytest.approx(0.0, abs=1e-12)
    assert ause(err, np.random.default_rng(1).random(500)) > 0.05


def test_sparsification_removes_the_worst_ranked_points_first():
    err = np.array([1.0, 1.0, 1.0, 10.0])
    curve = sparsification(err, err, np.array([0.0, 0.25]))
    assert curve[0] > 4.0 and curve[1] == pytest.approx(1.0)


def test_reliability_bins_follow_the_predicted_sigma():
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.001, 0.01, 4000)
    err = np.abs(rng.normal(scale=sigma))
    bins = reliability(err, sigma, n_bins=4)
    assert len(bins) == 4
    assert [b["observed_rms_m"] for b in bins] == sorted(b["observed_rms_m"] for b in bins)
    for b in bins:
        assert b["observed_rms_m"] == pytest.approx(b["predicted_sigma_m"], rel=0.2)
