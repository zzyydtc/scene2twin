import numpy as np
import pytest

from eval.colmap_io import quat_wxyz_to_matrix
from registration.manhattan import (
    estimate_normals, extent_fit, histogram_fit, manhattan_axes, occupancy_iou, register_manhattan,
    rotation_group_24,
)
from registration.sim3 import Sim3

RNG = np.random.default_rng(3)


def make_room(n=6000, size=(5.0, 4.0, 2.6), noise=0.005, clutter=True) -> np.ndarray:
    """Points on the six faces of a box, plus an asymmetric box-shaped obstacle."""
    lx, ly, lz = size
    pts = []
    for axis, extent in enumerate(size):
        others = [i for i in range(3) if i != axis]
        for face in (0.0, extent):
            p = np.zeros((n // 6, 3))
            p[:, axis] = face
            p[:, others[0]] = RNG.uniform(0, size[others[0]], len(p))
            p[:, others[1]] = RNG.uniform(0, size[others[1]], len(p))
            pts.append(p)
    if clutter:  # breaks the box's own symmetry
        p = np.zeros((1500, 3))
        p[:, 0] = RNG.uniform(0.2, 1.4, 1500)
        p[:, 1] = 0.3
        p[:, 2] = RNG.uniform(0.0, 0.9, 1500)
        pts.append(p)
        q = np.zeros((1000, 3))
        q[:, 0] = 4.2
        q[:, 1] = RNG.uniform(2.5, 3.9, 1000)
        q[:, 2] = RNG.uniform(0.0, 1.8, 1000)
        pts.append(q)
    return np.vstack(pts) + RNG.normal(scale=noise, size=(sum(len(p) for p in pts), 3))


def test_rotation_group_has_24_proper_rotations():
    group = rotation_group_24()
    assert len(group) == 24
    assert all(np.isclose(np.linalg.det(g), 1.0) for g in group)


def test_manhattan_axes_found_in_rotated_room():
    room = make_room()
    rot = quat_wxyz_to_matrix(RNG.normal(size=4))
    axes = manhattan_axes(estimate_normals(room @ rot.T, 0.1))
    # Each recovered axis must align (up to sign) with one rotated world axis.
    dots = np.abs(axes.T @ rot)  # rows: recovered axes, cols: true axes
    assert np.all(dots.max(axis=1) > np.cos(np.radians(5)))


def test_occupancy_iou_penalises_shrinking():
    room = make_room(clutter=False)
    assert occupancy_iou(room, room, 0.1) == pytest.approx(1.0)
    assert occupancy_iou(room * 0.1, room, 0.1) < 0.1


def test_extent_fit_recovers_scale_and_translation():
    room = make_room()
    truth = Sim3(0.4, np.eye(3), np.array([1.0, -2.0, 0.5]))
    fit = extent_fit(room, truth.apply(room), np.eye(3))
    assert fit.scale == pytest.approx(0.4, rel=0.02)
    np.testing.assert_allclose(fit.translation, truth.translation, atol=0.05)


def test_register_manhattan_recovers_known_placement():
    dst = make_room(n=12000)
    rot = quat_wxyz_to_matrix(RNG.normal(size=4))
    truth = Sim3(0.35, rot, np.array([3.0, 1.0, -2.0]))
    # Source: a sparse, noisier, different sampling of the same room in another frame.
    sparse = make_room(n=1500, noise=0.02)
    # Inverse of truth applied to the room gives a source whose registration to dst is `truth`.
    src = (sparse - truth.translation) @ rot / truth.scale
    res = register_manhattan(src, dst)
    err = np.linalg.norm(res.transform.apply(src) - sparse, axis=1)
    assert np.median(err) < 0.1
    assert res.transform.scale == pytest.approx(truth.scale, rel=0.05)


def test_histogram_fit_recovers_scale_and_shift_with_partial_overlap():
    room = make_room(n=12000)
    truth = Sim3(0.4, np.eye(3), np.array([1.0, -2.0, 0.5]))
    partial = room[room[:, 0] < 4.6]  # source misses the far wall but keeps its clutter
    fit, score = histogram_fit(partial, truth.apply(room), np.eye(3), init_scale=0.37)
    assert fit.scale == pytest.approx(0.4, rel=0.03)
    np.testing.assert_allclose(fit.translation, truth.translation, atol=0.06)
    assert score > 0.3
