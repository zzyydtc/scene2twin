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
