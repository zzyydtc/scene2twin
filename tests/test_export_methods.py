import numpy as np
from scipy.spatial.transform import Rotation

from eval.colmap_io import centre_from_pose
from scripts.export_methods import camera_vectors, place_cameras, to_view, trimmed_fit


def make_poses(n: int = 10, seed: int = 0) -> dict:
    """Camera-from-world poses (R, t) of n cameras spread around the origin, keyed by name."""
    rng = np.random.default_rng(seed)
    poses = {}
    for i in range(n):
        r = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
        centre = rng.normal(size=3) * 2.0
        poses[f"img{i:02d}.JPG"] = (r, -r @ centre)
    return poses


def move_camera(pose: tuple, offset: np.ndarray) -> tuple:
    r, t = pose
    return r, t - r @ offset  # the centre moves by ``offset``


def test_camera_vectors_follow_colmap_convention():
    r = np.eye(3)
    centre, forward, up = camera_vectors((r, np.array([1.0, 2.0, 3.0])))
    assert np.allclose(centre, [-1.0, -2.0, -3.0])
    assert np.allclose(forward, [0.0, 0.0, 1.0])
    assert np.allclose(up, [0.0, -1.0, 0.0])  # COLMAP's y axis points down


def test_to_view_applies_matrix_and_keeps_unit_directions():
    matrix = np.eye(4)
    matrix[:3, :3] = 2.0 * Rotation.from_euler("z", 90, degrees=True).as_matrix()
    matrix[:3, 3] = [1.0, 0.0, 0.0]
    entry = to_view(matrix, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(entry["position"], [1.0, 2.0, 0.0])  # scaled by 2, turned 90 degrees, then shifted
    assert np.isclose(np.linalg.norm(entry["forward"]), 1.0, atol=1e-3)
    assert np.allclose(entry["forward"], [0.0, 1.0, 0.0], atol=1e-3)


def test_exact_poses_have_zero_error_in_the_viewer_frame():
    ref = make_poses()
    cameras = place_cameras(ref, ref, np.eye(4), type("Sim", (), {"apply": lambda self, x: x, "rotation": np.eye(3)})())
    assert len(cameras) == len(ref)
    assert max(c["error_m"] for c in cameras) < 1e-9
    assert max(c["rotation_error_deg"] for c in cameras) < 1e-6


def test_trimmed_fit_leaves_out_the_cameras_that_were_moved():
    ref = make_poses(n=12)
    est = dict(ref)
    moved = ["img01.JPG", "img07.JPG"]
    for name in moved:
        est[name] = move_camera(ref[name], np.array([3.0, -2.0, 4.0]))
    sim, fitted = trimmed_fit(est, ref, fraction=0.8)
    assert not set(moved) & fitted
    assert len(fitted) == int(np.ceil(0.8 * 12))
    # With the moved cameras left out of the fit, the others sit on the reference.
    errors = [np.linalg.norm(sim.apply(centre_from_pose(*est[n])[None])[0] - centre_from_pose(*ref[n])) for n in fitted]
    assert max(errors) < 1e-6
    # Their error is still reported in full for the cameras that were left out.
    cameras = place_cameras(est, ref, np.eye(4), sim, fitted)
    by_name = {c["name"]: c for c in cameras}
    assert all(by_name[n]["error_m"] > 1.0 and not by_name[n]["fitted"] for n in moved)
