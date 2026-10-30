"""Global-to-fine registration of a reconstruction onto a reference cloud.

Two clouds with unknown relative scale are aligned in three stages:

1. Normalise both clouds (centre, RMS radius) so FPFH radii are comparable.
2. Coarse: FPFH features (Open3D) -> mutual nearest-neighbour matches in feature
   space -> RANSAC over minimal 3-point Sim(3) hypotheses (own implementation).
3. Fine: trimmed point-to-point ICP with scale (own implementation).

The result is a ``Sim3`` that maps the source cloud, in its original units, into
the frame of the destination cloud.
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from registration.sim3 import Sim3, umeyama


def normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre a cloud and scale it to unit RMS radius. Returns (cloud, centre, rms)."""
    centre = points.mean(axis=0)
    rms = float(np.sqrt(((points - centre) ** 2).sum(axis=1).mean()))
    return (points - centre) / rms, centre, rms


def compose(outer: Sim3, inner: Sim3) -> Sim3:
    """Sim3 equivalent to applying ``inner`` first, then ``outer``."""
    return Sim3(
        outer.scale * inner.scale,
        outer.rotation @ inner.rotation,
        outer.scale * outer.rotation @ inner.translation + outer.translation,
    )


def similarity(scale: float, centre: np.ndarray) -> Sim3:
    """Sim3 mapping ``x`` to ``(x - centre) / scale``."""
    return Sim3(1.0 / scale, np.eye(3), -centre / scale)


def inverse_similarity(scale: float, centre: np.ndarray) -> Sim3:
    """Sim3 mapping ``y`` to ``y * scale + centre`` (undoes ``similarity``)."""
    return Sim3(scale, np.eye(3), centre)


def fpfh(points: np.ndarray, voxel: float, radius_factor: float = 5.0):
    """Downsample and compute FPFH. Returns (points (N,3), features (N,33))."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * radius_factor, max_nn=100)
    )
    return np.asarray(pc.points), np.asarray(feat.data).T.copy()


def mutual_matches(feat_src: np.ndarray, feat_dst: np.ndarray) -> np.ndarray:
    """Mutual nearest neighbours in feature space. Returns (M, 2) index pairs."""
    _, fwd = cKDTree(feat_dst).query(feat_src)
    _, bwd = cKDTree(feat_src).query(feat_dst)
    src_idx = np.arange(len(feat_src))
    keep = bwd[fwd] == src_idx
    return np.stack([src_idx[keep], fwd[keep]], axis=1)


def ransac_sim3(
    src: np.ndarray, dst: np.ndarray, tau: float, iters: int = 20000,
    rng: np.random.Generator | None = None,
    scale_range: tuple[float, float] = (0.2, 5.0),
) -> tuple[Sim3, np.ndarray]:
    """RANSAC over putative correspondences ``src[i] <-> dst[i]``.

    Each hypothesis is a Sim(3) from 3 random pairs. Hypotheses whose scale is
    outside ``scale_range`` are rejected. The best model is refit on its inliers.
    Returns (transform, inlier mask).
    """
    rng = rng or np.random.default_rng(0)
    n = len(src)
    if n < 3:
        raise ValueError("need at least 3 correspondences")
    best_count, best_model = -1, None
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        try:
            model = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not (scale_range[0] <= model.scale <= scale_range[1]):
            continue
        resid = np.linalg.norm(model.apply(src) - dst, axis=1)
        count = int((resid < tau).sum())
        if count > best_count:
            best_count, best_model = count, model
    if best_model is None:
        raise RuntimeError("RANSAC found no valid hypothesis")
    inliers = np.linalg.norm(best_model.apply(src) - dst, axis=1) < tau
    if inliers.sum() >= 3:
        best_model = umeyama(src[inliers], dst[inliers])
        inliers = np.linalg.norm(best_model.apply(src) - dst, axis=1) < tau
    return best_model, inliers


def icp_sim3(
    src: np.ndarray, dst: np.ndarray, init: Sim3, max_dist: float,
    iters: int = 60, trim: float = 0.9, tol: float = 1e-7,
    scale_bounds: tuple[float, float] | None = None, tree: cKDTree | None = None,
) -> Sim3:
    """Trimmed point-to-point ICP with scale.

    Each iteration matches current source points to their nearest destination
    point, keeps pairs within ``max_dist`` and the closest ``trim`` fraction,
    and refits Sim(3) on the original source points (so error does not accumulate).

    Scale is a degenerate degree of freedom for one-sided nearest-neighbour
    objectives: shrinking the source onto any surface lowers the residual. When
    ``scale_bounds`` is given, the fitted scale is clamped to it.
    """
    tree = tree if tree is not None else cKDTree(dst)
    model = init
    prev = np.inf
    for _ in range(iters):
        moved = model.apply(src)
        d, j = tree.query(moved)
        keep = d < max_dist
        if keep.sum() < 3:
            break
        order = np.argsort(d[keep])
        keep_idx = np.flatnonzero(keep)[order[: max(3, int(trim * keep.sum()))]]
        a, b = src[keep_idx], dst[j[keep_idx]]
        model = umeyama(a, b)
        if scale_bounds is not None and not (scale_bounds[0] <= model.scale <= scale_bounds[1]):
            clamped = float(np.clip(model.scale, *scale_bounds))
            model = Sim3(clamped, model.rotation,
                         b.mean(axis=0) - clamped * model.rotation @ a.mean(axis=0))
        err = float(d[keep_idx].mean())
        if abs(prev - err) < tol:
            break
        prev = err
    return model


@dataclass
class RegistrationResult:
    transform: Sim3  # source (original units) -> destination frame
    n_matches: int
    n_inliers: int
    coarse_transform: Sim3


def register(
    src_points: np.ndarray, dst_points: np.ndarray, voxel: float = 0.06,
    tau: float | None = None, icp_dist: float | None = None, iters: int = 20000, seed: int = 0,
    scale_range: tuple[float, float] = (0.7, 1.4), icp_scale_slack: float = 0.15,
) -> RegistrationResult:
    """Align ``src_points`` to ``dst_points`` without any initial guess.

    ``voxel``, ``tau`` and ``icp_dist`` are in normalised units (unit RMS radius).
    ``scale_range`` bounds the relative scale of the coarse stage in those units. It
    encodes the assumption that both clouds cover a similar extent of the scene;
    without it the scale freedom lets RANSAC and ICP collapse the source onto a
    small patch of the destination.
    """
    src_n, c_s, r_s = normalise(src_points)
    dst_n, c_d, r_d = normalise(dst_points)

    p_s, f_s = fpfh(src_n, voxel)
    p_d, f_d = fpfh(dst_n, voxel)
    pairs = mutual_matches(f_s, f_d)
    tau = tau if tau is not None else 2.0 * voxel
    coarse_n, inl = ransac_sim3(
        p_s[pairs[:, 0]], p_d[pairs[:, 1]], tau, iters=iters, rng=np.random.default_rng(seed),
        scale_range=scale_range,
    )
    fine_n = icp_sim3(
        src_n, dst_n, coarse_n, max_dist=icp_dist if icp_dist is not None else 1.5 * voxel,
        scale_bounds=(coarse_n.scale * (1 - icp_scale_slack), coarse_n.scale * (1 + icp_scale_slack)),
    )

    # normalised -> original units: x_orig -> x_n -> (registered) -> y_n -> y_orig
    to_n = similarity(r_s, c_s)
    from_n = inverse_similarity(r_d, c_d)
    return RegistrationResult(
        transform=compose(from_n, compose(fine_n, to_n)),
        n_matches=len(pairs),
        n_inliers=int(inl.sum()),
        coarse_transform=compose(from_n, compose(coarse_n, to_n)),
    )
