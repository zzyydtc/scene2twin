import numpy as np
import pytest

from eval.colmap_io import quat_wxyz_to_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from registration.raster import (
    ReferenceFFT, box_sums, cross_correlate, icp_plane_sim3, occupancy, register_raster, surface_normals,
)
from registration.sim3 import Sim3
from tests.test_manhattan import make_room

RNG = np.random.default_rng(11)


def test_box_sums_match_brute_force():
    vol = RNG.random((5, 4, 3)).astype(np.float32)
    box = (2, 3, 2)
    out = box_sums(vol, box)
    padded = np.pad(vol, [(b - 1, b - 1) for b in box])
    expected = np.zeros(out.shape)
    for k in np.ndindex(out.shape):
        expected[k] = padded[k[0]:k[0] + box[0], k[1]:k[1] + box[1], k[2]:k[2] + box[2]].sum()
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_cross_correlate_peak_is_the_shift():
    b = np.zeros((20, 18, 12), dtype=np.float32)
    a = np.zeros((4, 5, 3), dtype=np.float32)
    a[1:3, 1:4, 1:2] = 1.0
    shift = (7, 9, 5)  # a[i] sits at b[i + shift]
    b[shift[0] + 1:shift[0] + 3, shift[1] + 1:shift[1] + 4, shift[2] + 1:shift[2] + 2] = 1.0
    ref = ReferenceFFT(b)
    fft_shape, fft_b = ref.get(a.shape)
    corr = cross_correlate(a, b, fft_b, fft_shape)
    assert corr.shape == tuple(x + y - 1 for x, y in zip(a.shape, b.shape))
    k = np.unravel_index(int(np.argmax(corr)), corr.shape)
    assert tuple(np.array(k) - (np.array(a.shape) - 1)) == shift
    assert corr[k] == pytest.approx(a.sum())


def test_occupancy_ignores_points_outside_the_grid():
    pts = np.array([[0.05, 0.05, 0.05], [9.0, 9.0, 9.0]])
    grid = occupancy(pts, np.zeros(3), 0.1, (5, 5, 5))
    assert grid[0, 0, 0] == 1.0 and grid.sum() > 1.0  # dilated around the first point only


def test_register_raster_recovers_a_partial_reconstruction():
    """The source misses the far part of the room, the case that defeats extent-based scaling."""
    dst = make_room(n=12000)
    rot = quat_wxyz_to_matrix(RNG.normal(size=4))
    truth = Sim3(0.35, rot, np.array([3.0, 1.0, -2.0]))
    sparse = make_room(n=1500, noise=0.02)
    sparse = sparse[sparse[:, 0] < 3.6]
    src = (sparse - truth.translation) @ rot / truth.scale
    res = register_raster(src, dst, scale_prior=(0.35, 0.1))
    err = np.linalg.norm(res.transform.apply(src) - sparse, axis=1)
    assert np.median(err) < 0.05
    assert res.transform.scale == pytest.approx(0.35, rel=0.03)


def test_point_to_plane_icp_removes_a_rotation_error():
    """A few degrees of rotation plus a shift, on sparse points from the walls of a room."""
    room = make_room()
    src = room[RNG.choice(len(room), 600, replace=False)]
    tree = cKDTree(room)
    normals = surface_normals(room, tree)
    start = Sim3(1.0, Rotation.from_euler("xyz", [4.0, -5.0, 6.0], degrees=True).as_matrix(), np.array([0.15, -0.1, 0.05]))
    out = icp_plane_sim3(src, room, normals, tree, start, (0.4, 0.2, 0.1, 0.05), (0.95, 1.05))
    assert np.degrees(Rotation.from_matrix(out.rotation).magnitude()) < 0.7  # the true model is the identity
    assert np.linalg.norm(out.apply(src) - src, axis=1).mean() < 0.02
