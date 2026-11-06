"""Translation and scale search by volumetric cross-correlation.

The Manhattan stage (registration/manhattan.py) fixes the rotation but sets scale and
translation from room extents, which breaks when the reconstruction covers only part of
the room. This module replaces that step with the approach used in image registration for
remote sensing: rasterise both clouds into occupancy grids and find the shift that
maximises their correlation with an FFT, for every candidate rotation and a grid of scales.

The score at a shift is the cosine similarity between the source occupancy ``A`` and the
reference occupancy ``B`` *restricted to the box that ``A`` covers*:

    score = sum(A * B) / sqrt(sum(A) * sum(B inside the box of A))

Restricting ``B`` to the source box is what lets a partial reconstruction match a larger
scan: scan structure outside the part of the room the cameras saw is not held against it.
The score is symmetric inside the box, so shrinking the source onto one surface does not
raise it the way a one-sided nearest-neighbour distance does.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
import open3d as o3d
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from registration.manhattan import (
    estimate_normals, manhattan_axes, occupancy_iou, rotation_group_24, voxel_keys,
)
from registration.pipeline import normalise
from registration.sim3 import Sim3


def occupancy(points: np.ndarray, origin: np.ndarray, voxel: float, shape: tuple[int, int, int]) -> np.ndarray:
    """Binary occupancy grid, dilated by one voxel so matches tolerate half-voxel offsets."""
    idx = np.floor((points - origin) / voxel).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    grid = np.zeros(shape, dtype=np.float32)
    idx = idx[ok]
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return maximum_filter(grid, size=3, mode="constant")


def box_sums(vol: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    """Sum of ``vol`` over a ``box``-shaped window at every offset, zero outside ``vol``.

    Output index ``k`` along an axis is the offset ``d = k - (box - 1)`` of the window's
    first voxel in ``vol`` coordinates, so the output has shape ``vol.shape + box - 1``.
    """
    pad = [(a - 1, a - 1) for a in box]
    integral = np.pad(vol, pad).astype(np.float32)  # counts stay far below 2**24, so exact
    for axis in range(3):
        integral = integral.cumsum(axis=axis, dtype=np.float32)
    integral = np.pad(integral, ((1, 0), (1, 0), (1, 0)))
    out = tuple(n + a - 1 for n, a in zip(vol.shape, box))
    ax, ay, az = box

    def corner(i: int, j: int, k: int) -> np.ndarray:
        return integral[i:i + out[0], j:j + out[1], k:k + out[2]]

    return (corner(ax, ay, az) - corner(0, ay, az) - corner(ax, 0, az) - corner(ax, ay, 0)
            + corner(0, 0, az) + corner(0, ay, 0) + corner(ax, 0, 0) - corner(0, 0, 0))


def cross_correlate(a: np.ndarray, b: np.ndarray, fft_b: np.ndarray, fft_shape: tuple[int, int, int]) -> np.ndarray:
    """``out[k] = sum_i a[i] * b[i + d]`` with ``d = k - (a.shape - 1)`` for every overlapping shift.

    ``fft_b`` is ``rfftn(b, fft_shape)``, computed once by the caller. The output has shape
    ``a.shape + b.shape - 1``.
    """
    fft_a = sfft.rfftn(a, fft_shape, workers=-1)
    corr = sfft.irfftn(fft_b * np.conj(fft_a), fft_shape, workers=-1)
    corr = np.roll(corr, shift=[s - 1 for s in a.shape], axis=(0, 1, 2))
    return corr[tuple(slice(0, an + bn - 1) for an, bn in zip(a.shape, b.shape))]


@dataclass
class RasterCandidate:
    rotation_index: int
    scale: float  # source units -> metres of the reference frame, before refinement
    score: float  # cosine similarity inside the source box, at the coarse grid
    transform: Sim3  # source (original units) -> reference frame, after refinement
    iou: float = 0.0
    mean_nn: float = 0.0


@dataclass
class RasterResult:
    transform: Sim3
    best: RasterCandidate
    candidates: list[RasterCandidate] = field(default_factory=list)


class ReferenceFFT:
    """FFT of the reference grid, cached per transform size (sizes repeat across nearby scales)."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}

    def get(self, box: tuple[int, int, int]) -> tuple[tuple[int, int, int], np.ndarray]:
        shape = tuple(sfft.next_fast_len(n + a - 1, real=True) for n, a in zip(self.grid.shape, box))
        if shape not in self._cache:
            self._cache[shape] = sfft.rfftn(self.grid, shape, workers=-1)
        return shape, self._cache[shape]


def search_rotation(
    src_c: np.ndarray, rotation: np.ndarray, scales: np.ndarray, ref: ReferenceFFT, origin_b: np.ndarray,
    voxel: float, min_box_ratio: float = 0.25,
) -> tuple[float, float, np.ndarray]:
    """Best (score, scale, translation) over ``scales`` for one rotation.

    ``src_c`` is the source centred on its mean, in original units; the returned translation
    moves ``scale * rotation @ src_c`` into the (centred) reference frame.
    """
    rotated = src_c @ rotation.T
    best = (-1.0, float(scales[0]), np.zeros(3))
    for s in scales:
        pts = rotated * s
        lo = np.percentile(pts, 0.5, axis=0)
        origin_a = lo - voxel
        hi = np.percentile(pts, 99.5, axis=0)
        shape_a = tuple(int(n) for n in np.ceil((hi - origin_a) / voxel) + 2)
        grid_a = occupancy(pts, origin_a, voxel, shape_a)
        mass_a = float(grid_a.sum())
        if mass_a < 10:
            continue
        fft_shape, fft_b = ref.get(shape_a)
        inter = cross_correlate(grid_a, ref.grid, fft_b, fft_shape)
        box_b = box_sums(ref.grid, shape_a)
        valid = box_b >= min_box_ratio * mass_a
        score = np.where(valid, inter / np.sqrt(mass_a * np.maximum(box_b, 1e-9)), 0.0)
        k = np.unravel_index(int(np.argmax(score)), score.shape)
        value = float(score[k])
        if value > best[0]:
            d = np.array(k) - (np.array(shape_a) - 1)
            best = (value, float(s), origin_b - origin_a + d * voxel)
    return best


def surface_normals(points: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Unit normals, from neighbourhoods of about four times the median point spacing (at least 6 cm)."""
    radius = max(0.06, 4.0 * float(np.median(tree.query(points[:: max(1, len(points) // 20000)], k=2)[0][:, 1])))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return np.asarray(pc.normals)


def icp_plane_sim3(
    src: np.ndarray, dst: np.ndarray, normals: np.ndarray, tree: cKDTree, init: Sim3,
    schedule: tuple[float, ...], scale_bounds: tuple[float, float], iters: int = 25, trim: float = 0.9,
) -> Sim3:
    """Trimmed point-to-plane ICP with scale, over a shrinking match distance.

    Each step solves the linearised least squares for a small rotation ``w``, translation and
    relative scale change that minimise the distance of the moved source points to the tangent
    planes of their nearest destination points. Unlike point-to-point, a point can slide along
    a wall without cost, which is what lets a few degrees of rotation error be removed from
    sparse points. The scale is clamped to ``scale_bounds``.
    """
    model = init
    for max_dist in schedule:
        for _ in range(iters):
            moved = model.apply(src)
            dist, j = tree.query(moved)
            n, q = normals[j], dst[j]
            residual = np.einsum("ij,ij->i", q - moved, n)
            idx = np.flatnonzero(dist < max_dist)
            if len(idx) < 20:
                break
            idx = idx[np.argsort(np.abs(residual[idx]))[: int(trim * len(idx))]]
            p, nn = moved[idx], n[idx]
            design = np.hstack([np.cross(p, nn), nn, np.einsum("ij,ij->i", nn, p)[:, None]])
            x = np.linalg.lstsq(design, residual[idx], rcond=None)[0]
            w, dt, ds = x[:3], x[3:6], x[6]
            rot = Rotation.from_rotvec(w).as_matrix()
            scale = float(np.clip(model.scale * (1 + ds), *scale_bounds))
            model = Sim3(scale, rot @ model.rotation, (scale / model.scale) * (rot @ model.translation) + dt)
            if np.linalg.norm(w) < 1e-5 and np.linalg.norm(dt) < 1e-5:
                break
    return model


def register_raster(
    src_points: np.ndarray, dst_points: np.ndarray, voxel: float = 0.15,
    scale_factor_range: tuple[float, float] = (0.5, 2.0), scale_step: float = 0.03,
    scale_prior: tuple[float, float] | None = None, n_refine: int = 6, icp_scale_slack: float = 0.04,
    normal_voxel: float = 0.03, iou_voxel: float = 0.1, seed: int = 0,
    icp_schedule: tuple[float, ...] = (0.4, 0.25, 0.15, 0.08, 0.04),
) -> RasterResult:
    """Align ``src_points`` (arbitrary frame and scale) to ``dst_points`` (metres).

    Rotation candidates are the 24 axis-aligned rotations between the two Manhattan frames.
    For each, scale and translation come from the correlation search above, searching scales
    on a grid around the RMS-radius ratio of the clouds (``scale_factor_range``), or, when a
    measured scale is known, ``scale_prior = (scale, relative_tolerance)`` restricts the grid
    to ``scale * (1 +- tolerance)``. The ``n_refine`` best rotations are refined with
    scale-bounded ICP and the winner is the one with the highest occupancy IoU, the same
    ranking the Manhattan stage uses. The refinement is point-to-plane ICP over the shrinking
    match distances in ``icp_schedule`` (``icp_plane_sim3``): the room axes are good to a few
    degrees only, and point-to-point ICP cannot remove the remainder from sparse points.
    """
    src_n, _, _ = normalise(src_points)
    dst_n, _, _ = normalise(dst_points)
    rng = np.random.default_rng(seed)
    ms = manhattan_axes(estimate_normals(src_n, normal_voxel), rng=rng)
    md = manhattan_axes(estimate_normals(dst_n, normal_voxel), rng=rng)

    src_mean, dst_mean = src_points.mean(axis=0), dst_points.mean(axis=0)
    src_c, dst_c = src_points - src_mean, dst_points - dst_mean
    rms_src = float(np.sqrt((src_c ** 2).sum(axis=1).mean()))
    rms_dst = float(np.sqrt((dst_c ** 2).sum(axis=1).mean()))
    s0 = rms_dst / rms_src
    if scale_prior is not None:
        centre, tol = scale_prior
        n = max(3, int(np.ceil(2 * tol / min(scale_step, 0.01))) + 1)
        scales = centre * np.linspace(1 - tol, 1 + tol, n)
    else:
        lo, hi = scale_factor_range
        n = int(np.ceil(np.log(hi / lo) / np.log(1 + 2 * scale_step))) + 1  # coarse; refined below
        scales = s0 * np.exp(np.linspace(np.log(lo), np.log(hi), n))

    # Reference grid. Scale is searched coarsely for every rotation, then finely around the best
    # coarse scale of the rotations worth refining.
    pad = 2 * voxel
    origin_b = dst_c.min(axis=0) - pad
    shape_b = tuple(int(n) for n in np.ceil((dst_c.max(axis=0) + pad - origin_b) / voxel) + 1)
    ref = ReferenceFFT(occupancy(dst_c, origin_b, voxel, shape_b))

    coarse = []
    for k, perm in enumerate(rotation_group_24()):
        rotation = md @ perm @ ms.T
        score, scale, t = search_rotation(src_c, rotation, scales, ref, origin_b, voxel)
        if score > 0:
            coarse.append((score, scale, t, k, rotation))
    if not coarse:
        raise RuntimeError("no rotation candidate produced a valid correlation")
    coarse.sort(key=lambda c: -c[0])

    candidates = []
    fine_grid = np.exp(np.linspace(-1, 1, 9) * np.log(1 + 2 * scale_step) * 1.0)
    for score, scale, t, k, rotation in coarse[:n_refine]:
        score, scale, t = search_rotation(src_c, rotation, scale * fine_grid, ref, origin_b, voxel)
        full = Sim3(scale, rotation, dst_mean + t - scale * rotation @ src_mean)
        candidates.append(RasterCandidate(k, scale, score, full))
    coarse = candidates

    tree = cKDTree(dst_points)
    dst_keys = voxel_keys(dst_points, iou_voxel)
    normals = surface_normals(dst_points, tree)
    refined = []
    for c in coarse:
        bounds = (c.scale * (1 - icp_scale_slack), c.scale * (1 + icp_scale_slack))
        fine = icp_plane_sim3(src_points, dst_points, normals, tree, c.transform, icp_schedule, bounds)
        moved = fine.apply(src_points)
        refined.append(RasterCandidate(
            c.rotation_index, c.scale, c.score, fine,
            occupancy_iou(moved, dst_points, iou_voxel, dst_keys), float(tree.query(moved)[0].mean()),
        ))
    best = max(refined, key=lambda c: c.iou)
    return RasterResult(best.transform, best, sorted(refined, key=lambda c: -c.iou))
