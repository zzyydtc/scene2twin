"""Loading camera poses and point clouds for evaluation.

Poses follow the COLMAP convention: ``cam_from_world = [R | t]``, so the camera
centre in world coordinates is ``C = -R^T t``.
"""
import re
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def centre_from_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return -rotation.T @ translation


def read_reference_poses(images_txt: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a COLMAP ``images.txt``. Returns {basename: (R_cw, t_cw)}."""
    poses = {}
    lines = [ln for ln in Path(images_txt).read_text().splitlines() if not ln.startswith("#")]
    # Two lines per image: pose line, then the 2D points line (which may be empty).
    for pose_line in lines[0::2]:
        parts = pose_line.split()
        if len(parts) < 10:
            continue
        quat = [float(v) for v in parts[1:5]]
        t = np.array([float(v) for v in parts[5:8]])
        name = Path(parts[9]).name
        poses[name] = (quat_wxyz_to_matrix(quat), t)
    return poses


def read_model_poses(model_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read poses from a COLMAP model folder (binary or text) via pycolmap."""
    rec = pycolmap.Reconstruction(str(model_dir))
    poses = {}
    for image in rec.images.values():
        m = image.cam_from_world().matrix()
        poses[Path(image.name).name] = (m[:3, :3].copy(), m[:3, 3].copy())
    return poses


def read_model_points(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz (N,3), reprojection error (N,), track length (N,)) of a model."""
    rec = pycolmap.Reconstruction(str(model_dir))
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype=np.float64)
    err = np.array([p.error for p in pts], dtype=np.float64)
    track = np.array([p.track.length() for p in pts], dtype=np.int64)
    return xyz, err, track


def read_meshlab_transforms(mlp_path: Path) -> dict[str, np.ndarray]:
    """Parse ``MLMatrix44`` blocks of a MeshLab project. Returns {filename: 4x4}."""
    text = Path(mlp_path).read_text()
    out = {}
    for m in re.finditer(
        r'<MLMesh[^>]*filename="([^"]+)"[^>]*>\s*<MLMatrix44>([^<]+)</MLMatrix44>', text
    ):
        values = np.array(m.group(2).split(), dtype=np.float64)
        out[m.group(1)] = values.reshape(4, 4)
    return out


def load_reference_scan(scan_dir: Path, voxel: float = 0.02) -> np.ndarray:
    """Load every scan listed in ``scan_alignment.mlp``, move it into the common
    frame and voxel-downsample. Returns an (N, 3) array."""
    scan_dir = Path(scan_dir)
    transforms = read_meshlab_transforms(scan_dir / "scan_alignment.mlp")
    clouds = []
    for filename, mat in transforms.items():
        pc = o3d.io.read_point_cloud(str(scan_dir / filename))
        pc.transform(mat)
        clouds.append(pc.voxel_down_sample(voxel))
    return np.vstack([np.asarray(pc.points) for pc in clouds])
