import numpy as np
import pytest

from eval.colmap_io import quat_wxyz_to_matrix
from semantics.scene_graph import build_scene_graph, scale_from_reference
from semantics.structure import OBJECT, camera_up, detect_planes, label_points, room_frame
from tests.test_manhattan import make_room

SIZE = (5.0, 4.0, 2.6)  # x, y, z of make_room; z is the vertical direction


def cameras(rot, n=6):
    """Capture positions well inside the SIZE room, in the rotated frame."""
    rng = np.random.default_rng(11)
    inside = rng.uniform([1.0, 1.0, 1.0], [4.0, 3.0, 1.8], size=(n, 3))
    return inside @ rot.T


def prepare(points, rot):
    """Rotate a z-up room into an arbitrary frame; cameras look along its horizontal."""
    pts = points @ rot.T
    cam_up = rot @ np.array([0.0, 0.0, 1.0])
    # A COLMAP camera with its y axis pointing down: R_cw maps world->camera.
    return pts, cam_up


def test_scale_from_reference():
    assert scale_from_reference([0, 0, 0], [2, 0, 0], 0.5) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        scale_from_reference([0, 0, 0], [0, 0, 0], 1.0)


def test_camera_up_from_upright_cameras():
    # Camera frame: x right, y down, z forward; looking along world +x with world z up.
    r_wc = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # camera->world
    up = camera_up([r_wc.T])
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)


def test_planes_and_labels_on_rotated_room():
    rng = np.random.default_rng(5)
    room = make_room(n=9000)
    rot = quat_wxyz_to_matrix(rng.normal(size=4))
    pts, cam_up = prepare(room, rot)
    frame = room_frame(pts, cam_up)
    planes = detect_planes(pts, frame, cameras(rot))
    roles = sorted(p.role for p in planes)
    assert roles.count("floor") == 1 and roles.count("ceiling") == 1 and roles.count("wall") == 4
    floor = next(p for p in planes if p.role == "floor")
    ceiling = next(p for p in planes if p.role == "ceiling")
    assert abs(ceiling.offset - floor.offset) == pytest.approx(SIZE[2], abs=0.05)

    labels, conf = label_points(pts, frame, planes)
    # 9000 points lie on the six room faces and 2500 belong to the clutter objects.
    on_plane = (labels != OBJECT).mean()
    assert on_plane == pytest.approx(9000 / 11500, abs=0.03)
    assert conf[labels != OBJECT].mean() > 0.6


def test_missing_ceiling_is_not_reported():
    rng = np.random.default_rng(6)
    room = make_room(n=9000, clutter=False)
    room = room[room[:, 2] < SIZE[2] - 0.2]  # drop the ceiling points
    rot = quat_wxyz_to_matrix(rng.normal(size=4))
    pts, cam_up = prepare(room, rot)
    frame = room_frame(pts, cam_up)
    planes = detect_planes(pts, frame, cameras(rot))
    assert not any(p.role == "ceiling" for p in planes)
    labels, conf = label_points(pts, frame, planes)
    graph = build_scene_graph(pts, labels, conf, frame, planes)
    ceiling = [n for n in graph["nodes"] if n["id"] == "ceiling"]
    assert ceiling and ceiling[0]["status"] == "not_observed"


def test_scene_graph_has_object_on_floor_and_metric_scale():
    rng = np.random.default_rng(7)
    room = make_room(n=9000)
    rot = quat_wxyz_to_matrix(rng.normal(size=4))
    pts, cam_up = prepare(room, rot)
    frame = room_frame(pts, cam_up)
    planes = detect_planes(pts, frame, cameras(rot))
    labels, conf = label_points(pts, frame, planes)
    graph = build_scene_graph(pts, labels, conf, frame, planes, scale=2.0, unit="m")
    assert graph["unit"] == "m"
    assert sorted(graph["room_extent"])[-1] == pytest.approx(2.0 * max(SIZE), rel=0.05)
    assert any(e["relation"] == "perpendicular" for e in graph["edges"])
