import numpy as np
import pytest

from scripts.export_viewer import room_to_view


@pytest.mark.parametrize("up", [0, 1, 2])
def test_vertical_axis_becomes_y(up):
    coords = np.zeros((1, 3))
    coords[0, up] = 1.0
    assert np.allclose(room_to_view(coords, up), [[0.0, 1.0, 0.0]])


@pytest.mark.parametrize("up", [0, 1, 2])
def test_handedness_is_preserved(up):
    """The axis permutation must be cyclic: a mirrored room would read as a different room."""
    basis = np.eye(3)
    mapped = room_to_view(basis, up)
    assert np.linalg.det(mapped) == pytest.approx(1.0)


def test_scan_to_view_matrix_matches_the_point_mapping():
    """A scan point that coincides with a placed reconstruction point must land on that point's viewer position."""
    from scipy.spatial.transform import Rotation

    from registration.sim3 import Sim3
    from scripts.export_viewer import scan_to_view_matrix

    rng = np.random.default_rng(3)
    placement = Sim3(0.37, Rotation.from_euler("xyz", [10, -40, 75], degrees=True).as_matrix(), np.array([2.0, -1.0, 0.5]))
    axes = Rotation.from_euler("xyz", [5, 20, -30], degrees=True).as_matrix()

    def to_view(x):
        return room_to_view(x @ axes, 1) * 0.5 - np.array([0.1, 0.2, 0.3])

    recon = rng.normal(size=(20, 3))
    m = scan_to_view_matrix(placement, to_view)
    moved = placement.apply(recon) @ m[:3, :3].T + m[:3, 3]
    assert np.allclose(moved, to_view(recon), atol=1e-9)
