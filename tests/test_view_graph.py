import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from sfm.view_graph import PAIR_ID_BASE, cycle_support, image_pair


def _graph(n=8, seed=0):
    """Consistent relative rotations of n cameras: R(i, j) takes camera i to camera j."""
    rng = np.random.default_rng(seed)
    poses = [Rotation.random(random_state=rng.integers(1 << 31)).as_matrix() for _ in range(n)]
    return {(i, j): poses[j] @ poses[i].T for i, j in itertools.combinations(range(n), 2)}


def test_image_pair_decodes_colmap_pair_ids():
    assert image_pair(3 * PAIR_ID_BASE + 7) == (3, 7)
    assert image_pair(1 * PAIR_ID_BASE + 2) == (1, 2)


def test_consistent_rotations_have_full_support():
    support = cycle_support(_graph())
    assert all(s == 1.0 for s in support.values())


def test_one_wrong_pair_has_low_support_and_barely_hurts_the_others():
    rotations = _graph()
    rotations[(2, 5)] = rotations[(2, 5)] @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
    support = cycle_support(rotations)
    assert support[(2, 5)] == 0.0
    # Every other pair loses at most the one loop that runs through the wrong pair.
    others = [s for pair, s in support.items() if pair != (2, 5)]
    assert min(others) >= 1 - 1 / 6 - 1e-9
    # Only the 12 pairs that share an image with (2, 5) lose that one loop; the other 15 keep all of theirs.
    assert sum(s == 1.0 for s in others) == 15


def test_small_noise_is_inside_the_tolerance():
    rng = np.random.default_rng(1)
    rotations = {k: Rotation.from_rotvec(rng.normal(scale=np.radians(1.0), size=3)).as_matrix() @ v for k, v in _graph().items()}
    assert min(cycle_support(rotations, tolerance_deg=10).values()) == 1.0


def test_pair_without_a_common_neighbour_is_not_penalised():
    rotations = {(0, 1): np.eye(3), (2, 3): Rotation.from_euler("x", 80, degrees=True).as_matrix()}
    assert cycle_support(rotations) == {(0, 1): 1.0, (2, 3): 1.0}


def test_reversed_pair_keys_are_handled():
    # A pair stored as (j, i) with i < j elsewhere must be read as the transpose.
    rotations = _graph(4)
    rotations[(3, 1)] = rotations.pop((1, 3)).T
    assert all(s == 1.0 for s in cycle_support(rotations).values())
