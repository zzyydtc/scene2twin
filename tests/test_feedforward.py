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


def test_cam_to_world_inverts_to_camera_from_world():
    from feedforward.mast3r_run import cam_to_world_to_extrinsic

    rot = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    cam2w = np.eye(4)
    cam2w[:3, :3], cam2w[:3, 3] = rot, [1.0, 2.0, 3.0]
    ext = cam_to_world_to_extrinsic(cam2w[None])[0]
    # A world point at the camera centre maps to the camera origin.
    assert np.allclose(ext[:, :3] @ np.array([1.0, 2.0, 3.0]) + ext[:, 3], 0.0)
    assert np.allclose(ext[:, :3], rot.T)


def test_network_focal_is_scaled_by_the_width_ratio():
    from feedforward.mast3r_run import focal_in_photo_pixels

    assert focal_in_photo_pixels(500.0, 512, 6048) == 500.0 * 6048 / 512


def test_network_size_matches_the_mast3r_loader_for_the_eth3d_photos():
    from feedforward.mast3r_run import network_size_for

    # 6048 x 4032 shrinks to 512 x 341 and is then cropped to a multiple of 16 in each direction.
    assert network_size_for((6048, 4032)) == (512, 336)
    assert network_size_for((4032, 6048)) == (336, 512)


def test_inside_box_drops_far_and_non_finite_points():
    from feedforward.mast3r_run import inside_box

    reference = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    points = np.array([[5.0, 5.0, 5.0], [14.0, 5.0, 5.0], [16.0, 5.0, 5.0], [np.nan, 0.0, 0.0], [1e9, 0.0, 0.0]])
    kept = inside_box(points, reference, margin=5.0)
    assert kept.tolist() == [[5.0, 5.0, 5.0], [14.0, 5.0, 5.0]]
