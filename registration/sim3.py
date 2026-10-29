"""Similarity (Sim(3)) alignment between two corresponding 3D point sets.

Implements the closed-form Umeyama solution: find scale s, rotation R and
translation t minimising sum_i || dst_i - (s * R @ src_i + t) ||^2.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the transform to an (N, 3) array of points."""
        return self.scale * points @ self.rotation.T + self.translation

    def matrix(self) -> np.ndarray:
        """4x4 homogeneous matrix (scale folded into the linear part)."""
        m = np.eye(4)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation
        return m


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares Sim(3) mapping ``src`` onto ``dst`` (both (N, 3), N >= 3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both have shape (N, 3)")
    n = src.shape[0]
    if n < 3:
        raise ValueError("need at least 3 correspondences")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = dst_c.T @ src_c / n
    u, d, vt = np.linalg.svd(cov)
    # Reflection guard: force a proper rotation (det = +1).
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1.0
    rotation = u @ s_fix @ vt

    var_src = (src_c ** 2).sum() / n
    scale = float(np.trace(np.diag(d) @ s_fix) / var_src) if with_scale else 1.0
    translation = mu_dst - scale * rotation @ mu_src
    return Sim3(scale, rotation, translation)
