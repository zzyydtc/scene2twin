"""Coarse registration for rooms from their dominant orthogonal directions.

Man-made interiors are dominated by three mutually orthogonal plane families
(floor/ceiling, two wall directions). Instead of matching local descriptors, this
module estimates those three axes in each cloud and enumerates the 24 proper
axis-aligned rotations relating the two frames. For each candidate, scale and
translation come from the room extents; candidates are refined with bounded ICP
and ranked by a symmetric voxel-occupancy overlap, which, unlike a one-sided
nearest-neighbour distance, cannot be improved by shrinking the source.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from registration.pipeline import (
    compose, icp_sim3, inverse_similarity, normalise, similarity,
)
from registration.sim3 import Sim3


def estimate_normals(points: np.ndarray, voxel: float) -> np.ndarray:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    return np.asarray(pc.normals)


def _best_direction(normals: np.ndarray, candidates: np.ndarray, cos_tol: float) -> np.ndarray:
    """Candidate direction supported by the most normals (sign-agnostic), refined."""
    support = (np.abs(normals @ candidates.T) > cos_tol).sum(axis=0)
    best = candidates[int(np.argmax(support))]
    for _ in range(3):  # refine: average of supporting normals with consistent sign
        dots = normals @ best
        sel = np.abs(dots) > cos_tol
        if sel.sum() < 3:
            break
        signed = normals[sel] * np.sign(dots[sel])[:, None]
        best = signed.mean(axis=0)
        best /= np.linalg.norm(best)
    return best


def manhattan_axes(
    normals: np.ndarray, angle_tol_deg: float = 8.0, n_candidates: int = 3000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Three orthonormal dominant directions (columns), right-handed.

    The order and signs are arbitrary; callers must not rely on them.
    """
    rng = rng or np.random.default_rng(0)
    cos_tol = np.cos(np.radians(angle_tol_deg))
    sin_tol = np.sin(np.radians(angle_tol_deg))
    pick = rng.choice(len(normals), size=min(n_candidates, len(normals)), replace=False)
    a1 = _best_direction(normals, normals[pick], cos_tol)

    perp = normals[np.abs(normals @ a1) < sin_tol * 1.5]
    if len(perp) < 10:
        raise ValueError("not enough surfaces perpendicular to the dominant direction")
    perp = perp - np.outer(perp @ a1, a1)
    norms = np.linalg.norm(perp, axis=1)
    perp = perp[norms > 1e-6] / norms[norms > 1e-6, None]
    pick = rng.choice(len(perp), size=min(n_candidates, len(perp)), replace=False)
    a2 = _best_direction(perp, perp[pick], cos_tol)
    a2 = a2 - (a2 @ a1) * a1
    a2 /= np.linalg.norm(a2)
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=1)


def rotation_group_24() -> list[np.ndarray]:
    """All signed permutation matrices with determinant +1."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            p = np.zeros((3, 3))
            for row, (col, sgn) in enumerate(zip(perm, signs)):
                p[row, col] = sgn
            if np.isclose(np.linalg.det(p), 1.0):
                out.append(p)
    return out


def extent_fit(src: np.ndarray, dst: np.ndarray, rotation: np.ndarray, pct: float = 2.0) -> Sim3:
    """Scale and translation after rotation, from robust per-axis extents.

    Scale is the median over the three axes of the extent ratio; translation
    matches the centres of the two robust bounding boxes.
    """
    rs = src @ rotation.T
    lo_s, hi_s = np.percentile(rs, [pct, 100 - pct], axis=0)
    lo_d, hi_d = np.percentile(dst, [pct, 100 - pct], axis=0)
    scale = float(np.median((hi_d - lo_d) / np.maximum(hi_s - lo_s, 1e-9)))
    centre_s, centre_d = (lo_s + hi_s) / 2, (lo_d + hi_d) / 2
    return Sim3(scale, rotation, centre_d - scale * centre_s)


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    """Unique occupied voxels as packed int64 keys."""
    idx = np.floor(points / voxel).astype(np.int64)
    idx -= np.array([-512, -512, -512])  # keep indices positive for packing
    return np.unique((idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2])


def occupancy_iou(a: np.ndarray, b: np.ndarray, voxel: float, keys_b: np.ndarray | None = None) -> float:
    """Intersection over union of occupied voxels. Symmetric, so it cannot be
    raised by shrinking one cloud onto a patch of the other. ``keys_b`` lets a
    caller reuse the voxelisation of the reference cloud."""
    ka = voxel_keys(a, voxel)
    kb = voxel_keys(b, voxel) if keys_b is None else keys_b
    inter = len(np.intersect1d(ka, kb, assume_unique=True))
    union = len(ka) + len(kb) - inter
    return inter / union if union else 0.0


def histogram_fit(
    src: np.ndarray, dst: np.ndarray, rotation: np.ndarray, init_scale: float,
    rel_range: float = 0.2, n_scales: int = 81, bin_size: float = 0.01, smooth_bins: float = 2.0,
) -> tuple[Sim3, float]:
    """Scale and translation from wall positions along the three room axes.

    After rotating the source into the destination's axes, planes appear as peaks
    in each axis's coordinate histogram. For every candidate scale, the shift per
    axis is the lag that maximises the zero-mean cross-correlation of the two
    smoothed histograms; the scale with the best summed correlation wins. Partial
    overlap is tolerated because only matching peaks contribute.
    Returns (transform, summed correlation score).
    """
    rs = src @ rotation.T
    pad = 1.0
    hist_d, y0 = [], []
    for a in range(3):
        lo = dst[:, a].min() - pad
        edges = np.arange(lo, dst[:, a].max() + pad + bin_size, bin_size)
        h = gaussian_filter1d(np.histogram(dst[:, a], bins=edges)[0].astype(float), smooth_bins)
        hist_d.append(h - h.mean())
        y0.append(lo)

    best = (-np.inf, None)
    for scale in np.linspace(init_scale * (1 - rel_range), init_scale * (1 + rel_range), n_scales):
        total, shifts = 0.0, np.zeros(3)
        for a in range(3):
            x = scale * rs[:, a]
            x0 = x.min()
            edges = np.arange(x0, x.max() + bin_size, bin_size)
            hs = gaussian_filter1d(np.histogram(x, bins=edges)[0].astype(float), smooth_bins)
            hs = hs - hs.mean()
            cc = np.correlate(hist_d[a], hs, mode="full")
            cc = cc / (np.linalg.norm(hist_d[a]) * np.linalg.norm(hs) + 1e-12)
            k = int(np.argmax(cc))
            total += cc[k]
            shifts[a] = (y0[a] - x0) + (k - (len(hs) - 1)) * bin_size
        if total > best[0]:
            best = (total, Sim3(float(scale), rotation, shifts))
    return best[1], float(best[0])


@dataclass
class Candidate:
    index: int
    transform: Sim3  # normalised units
    iou: float
    mean_nn: float
    hist_score: float = 0.0


@dataclass
class ManhattanResult:
    transform: Sim3  # source (original units) -> destination frame
    best: Candidate
    candidates: list[Candidate] = field(default_factory=list)


def register_manhattan(
    src_points: np.ndarray, dst_points: np.ndarray, normal_voxel: float = 0.03,
    iou_voxel: float = 0.1, scale_slack: float = 0.05, seed: int = 0,
    init_mode: str = "extent",
) -> ManhattanResult:
    """Align src to dst using Manhattan frames; units: normalised internally.

    ``init_mode`` selects the per-candidate starting scale/translation: ``"extent"``
    (robust bounding boxes) or ``"histogram"`` (per-axis wall-peak correlation).
    """
    if init_mode not in ("extent", "histogram"):
        raise ValueError("init_mode must be 'extent' or 'histogram'")
    src_n, c_s, r_s = normalise(src_points)
    dst_n, c_d, r_d = normalise(dst_points)
    rng = np.random.default_rng(seed)

    ms = manhattan_axes(estimate_normals(src_n, normal_voxel), rng=rng)
    md = manhattan_axes(estimate_normals(dst_n, normal_voxel), rng=rng)

    dst_tree = cKDTree(dst_n)
    dst_keys = voxel_keys(dst_n, iou_voxel)
    candidates = []
    for k, perm in enumerate(rotation_group_24()):
        rotation = md @ perm @ ms.T
        prior = extent_fit(src_n, dst_n, rotation)
        if not (0.4 <= prior.scale <= 2.5):
            continue
        if init_mode == "histogram":
            init, hist_score = histogram_fit(src_n, dst_n, rotation, prior.scale)
        else:
            init, hist_score = prior, 0.0
        bounds = (init.scale * (1 - scale_slack), init.scale * (1 + scale_slack))
        fine = icp_sim3(src_n, dst_n, init, max_dist=0.05, scale_bounds=bounds, tree=dst_tree)
        moved = fine.apply(src_n)
        candidates.append(Candidate(
            k, fine, occupancy_iou(moved, dst_n, iou_voxel, dst_keys),
            float(dst_tree.query(moved)[0].mean()), hist_score,
        ))
    if not candidates:
        raise RuntimeError("no candidate rotation gave a plausible scale")
    best = max(candidates, key=lambda c: c.iou)
    full = compose(inverse_similarity(r_d, c_d), compose(best.transform, similarity(r_s, c_s)))
    return ManhattanResult(full, best, sorted(candidates, key=lambda c: -c.iou))
