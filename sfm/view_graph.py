"""Remove wrongly verified image pairs from a COLMAP database before global mapping.

Global SfM averages the relative rotations of all verified pairs, so a few confidently wrong pairs
can flip a whole group of cameras (see docs/day8-sfm-engines.md). A pair is checked here without any
reference: its relative rotation must agree with what the other pairs say about the same two images.
For images i, j and any third image k that is paired with both, the loop i -> j -> k -> i has to
close, ``R_jk R_ij R_ik^T = I``. A wrong pair closes few of its loops.
"""
import shutil
from pathlib import Path

import numpy as np
import pycolmap

PAIR_ID_BASE = 2147483647  # COLMAP's pair id is image_id1 * 2147483647 + image_id2


def image_pair(pair_id: int) -> tuple[int, int]:
    j = pair_id % PAIR_ID_BASE
    return (pair_id - j) // PAIR_ID_BASE, j


def rotation_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def cycle_support(rotations: dict, tolerance_deg: float = 10.0) -> dict:
    """Fraction of loops through each pair that close within ``tolerance_deg``.

    ``rotations`` maps (i, j) to the rotation of image j relative to image i. Pairs that share no
    third image with a rotation get support 1.0: nothing contradicts them.
    """
    neighbours: dict = {}
    for a, b in rotations:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)

    def rel(a, b):
        return rotations[(a, b)] if (a, b) in rotations else rotations[(b, a)].T

    support = {}
    for i, j in rotations:
        errors = [rotation_angle_deg(rel(j, k) @ rel(i, j) @ rel(i, k).T) for k in neighbours[i] & neighbours[j]]
        support[(i, j)] = float(np.mean(np.array(errors) < tolerance_deg)) if errors else 1.0
    return support


def relative_rotations(database: Path, max_error_px: float = 4.0) -> dict:
    """Relative rotation of every verified pair, from RANSAC on its inlier matches.

    Uses the database's own cameras (the focal length COLMAP started from), so no reference or
    calibration is involved.
    """
    ransac = pycolmap.RANSACOptions()
    ransac.max_error = max_error_px
    db = pycolmap.Database.open(str(database))
    try:
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        camera_of = {im.image_id: cameras[im.camera_id] for im in db.read_all_images()}
        keypoints = {i: np.asarray(db.read_keypoints(i), dtype=np.float64)[:, :2] for i in camera_of}
        pair_ids, geometries = db.read_two_view_geometries()
        rotations = {}
        for pid, geometry in zip(pair_ids, geometries):
            i, j = image_pair(pid)
            m = np.asarray(geometry.inlier_matches, dtype=np.int64)
            if len(m) < 8:
                continue
            pose = pycolmap.estimate_relative_pose(
                camera_of[i], np.ascontiguousarray(keypoints[i][m[:, 0]]),
                camera_of[j], np.ascontiguousarray(keypoints[j][m[:, 1]]), ransac)
            if pose is not None:
                rotations[(i, j)] = pose["cam2_from_cam1"].rotation.matrix()
    finally:
        db.close()
    return rotations


def drop_inconsistent_pairs(database: Path, min_support: float = 0.5, tolerance_deg: float = 10.0) -> list[tuple[int, int]]:
    """Delete from ``database`` (in place) every pair whose loop support is below ``min_support``.

    Returns the dropped (image_id1, image_id2) pairs.
    """
    support = cycle_support(relative_rotations(database), tolerance_deg)
    dropped = [pair for pair, s in support.items() if s < min_support]
    db = pycolmap.Database.open(str(database))
    try:
        for i, j in dropped:
            db.delete_two_view_geometry(i, j)
    finally:
        db.close()
    return dropped
