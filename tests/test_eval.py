import numpy as np
import pytest

from eval.colmap_io import centre_from_pose, quat_wxyz_to_matrix, read_meshlab_transforms
from eval.metrics import accuracy_completeness, pose_errors
from registration.sim3 import umeyama

RNG = np.random.default_rng(0)


def random_rotation() -> np.ndarray:
    q = RNG.normal(size=4)
    return quat_wxyz_to_matrix(q)


def test_umeyama_recovers_known_transform():
    src = RNG.normal(size=(50, 3))
    r, s, t = random_rotation(), 2.5, np.array([1.0, -2.0, 0.5])
    dst = s * src @ r.T + t
    sim = umeyama(src, dst)
    assert sim.scale == pytest.approx(s)
    np.testing.assert_allclose(sim.rotation, r, atol=1e-9)
    np.testing.assert_allclose(sim.translation, t, atol=1e-9)


def test_umeyama_returns_proper_rotation_for_planar_points():
    src = RNG.normal(size=(30, 3))
    src[:, 2] = 0.0
    sim = umeyama(src, src.copy())
    assert np.linalg.det(sim.rotation) == pytest.approx(1.0)


def make_poses(n=8):
    poses = {}
    for i in range(n):
        poses[f"img{i}.jpg"] = (random_rotation(), RNG.normal(size=3))
    return poses


def apply_world_sim3(poses, s, r, t):
    """Express the same cameras in a world frame changed by X' = s R X + t."""
    out = {}
    for name, (r_cw, t_cw) in poses.items():
        c = centre_from_pose(r_cw, t_cw)
        c_new = s * r @ c + t
        r_cw_new = r_cw @ r.T
        out[name] = (r_cw_new, -r_cw_new @ c_new)
    return out


def test_pose_errors_zero_after_pure_similarity():
    ref = make_poses()
    est = apply_world_sim3(ref, 0.3, random_rotation(), np.array([4.0, 1.0, -3.0]))
    rep = pose_errors(est, ref)
    assert rep.ate_rmse_m < 1e-9
    assert rep.rotation_err_max_deg < 1e-4  # arccos near 0 is noisy at ~1e-6 deg
    assert rep.scale == pytest.approx(1 / 0.3)


def test_pose_errors_detects_perturbation():
    ref = make_poses(12)
    est = {k: (r, t.copy()) for k, (r, t) in ref.items()}
    r0, t0 = est["img0.jpg"]
    c = centre_from_pose(r0, t0) + np.array([0.5, 0.0, 0.0])
    est["img0.jpg"] = (r0, -r0 @ c)
    rep = pose_errors(est, ref)
    assert rep.ate_max_m > 0.2


def test_pose_errors_needs_three_common_images():
    ref = make_poses(5)
    est = {k: v for k, v in list(ref.items())[:2]}
    with pytest.raises(ValueError):
        pose_errors(est, ref)


def test_accuracy_completeness_known_offset():
    ref = RNG.uniform(size=(2000, 3))
    est = ref + np.array([0.03, 0.0, 0.0])
    res = accuracy_completeness(est, ref, thresholds=(0.05,))
    # Nearest-neighbour distance can only be <= the 0.03 shift.
    assert res["accuracy_mean_m"] <= 0.03 + 1e-9
    assert res["at_threshold"]["0.05"]["precision"] == pytest.approx(1.0)
    assert res["at_threshold"]["0.05"]["recall"] == pytest.approx(1.0)


def test_accuracy_completeness_penalises_missing_region():
    ref = RNG.uniform(size=(3000, 3))
    est = ref[ref[:, 0] < 0.5]
    res = accuracy_completeness(est, ref, thresholds=(0.02,))
    assert res["at_threshold"]["0.02"]["precision"] == pytest.approx(1.0)
    assert res["at_threshold"]["0.02"]["recall"] < 0.7


def test_meshlab_transform_parsing(tmp_path):
    mlp = tmp_path / "a.mlp"
    mlp.write_text(
        '<MLMesh label="s.ply" filename="s.ply">\n<MLMatrix44>\n'
        "1 0 0 1\n0 1 0 2\n0 0 1 3\n0 0 0 1\n</MLMatrix44>\n</MLMesh>"
    )
    mats = read_meshlab_transforms(mlp)
    np.testing.assert_allclose(mats["s.ply"][:3, 3], [1, 2, 3])
