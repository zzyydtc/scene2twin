import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from feedforward.refine import initial_model_text
from feedforward.vggt_run import confident_points, extrinsics_to_poses, focal_in_original_pixels


def test_extrinsics_become_named_rotation_translation_pairs():
    ext = np.stack([np.hstack([np.eye(3), [[1.0], [2.0], [3.0]]]), np.hstack([np.eye(3), np.zeros((3, 1))])])
    poses = extrinsics_to_poses(["a.jpg", "b.jpg"], ext)
    assert set(poses) == {"a.jpg", "b.jpg"}
    assert np.allclose(poses["a.jpg"][1], [1, 2, 3]) and np.allclose(poses["b.jpg"][0], np.eye(3))


def test_focal_is_scaled_per_axis_from_network_to_photo():
    intrinsic = np.zeros((3, 3, 3))
    intrinsic[:, 0, 0], intrinsic[:, 1, 1] = 300.0, 200.0
    fx, fy = focal_in_original_pixels(intrinsic, (500, 400), (1000, 1200))
    assert (fx, fy) == (600.0, 600.0)


def test_confident_points_keeps_the_top_fraction_and_drops_non_finite():
    points = np.arange(12, dtype=float).reshape(1, 2, 2, 3)
    conf = np.array([[[1.0, 2.0], [3.0, np.nan]]])
    kept = confident_points(points, conf, keep_fraction=0.5, max_points=100)
    assert len(kept) == 2  # of the 3 finite points the two most confident
    assert {tuple(p) for p in kept} == {(3.0, 4.0, 5.0), (6.0, 7.0, 8.0)}


def test_confident_points_thins_to_the_budget():
    points = np.random.default_rng(0).normal(size=(2, 10, 10, 3))
    conf = np.random.default_rng(1).uniform(size=(2, 10, 10))
    assert len(confident_points(points, conf, 1.0, max_points=50)) == 50


def test_initial_model_text_round_trips_through_colmap(tmp_path):
    rot = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    ext = np.stack([np.hstack([rot, [[0.1], [0.2], [0.3]]]), np.hstack([np.eye(3), [[0.0], [0.0], [1.0]]])])
    cameras, images, points = initial_model_text(["a.jpg", "b.jpg"], ext, {"a.jpg": 7, "b.jpg": 9}, 4, (600, 400), (500.0, 510.0))
    for name, body in zip(("cameras.txt", "images.txt", "points3D.txt"), (cameras, images, points)):
        (tmp_path / name).write_text(body)
    rec = pycolmap.Reconstruction(str(tmp_path))
    assert rec.num_reg_images() == 2 and rec.num_points3D() == 0
    cam = rec.cameras[4]
    assert cam.width == 600 and np.allclose(cam.params[:4], [500.0, 510.0, 300.0, 200.0])
    m = rec.images[7].cam_from_world().matrix()
    assert np.allclose(m[:3, :3], rot, atol=1e-9) and np.allclose(m[:3, 3], [0.1, 0.2, 0.3])
