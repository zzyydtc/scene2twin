"""Lift 2D instance masks (SAM) onto sparse 3D points, with cross-view confidence.

SAM is class-agnostic: it separates things but does not name them. Each SfM point
is observed in several images; in every image it falls inside some mask. Points
that repeatedly share a mask across views are grouped into an instance, and the
agreement between views is the instance's confidence.

Steps per image: keep masks smaller than ``max_area_frac`` of the image (larger ones
are room surfaces, handled by the plane stage), paint them from large to small so
the smallest mask containing a pixel wins, and read the id under each observation.
Across images: count how often two points share a mask, link pairs that share one in
at least ``min_views`` views and in at least ``min_ratio`` of the views that
observe both, and take connected components.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def assign_mask_ids(
    masks: list[dict], shape: tuple[int, int], xy: np.ndarray, max_area_frac: float = 0.25
) -> np.ndarray:
    """Mask id under each (x, y) pixel position, -1 where no object-sized mask applies.

    ``masks`` follow SAM's output (dicts with ``segmentation`` and ``area``). Larger
    masks are painted first so the smallest containing mask is the one read back.
    """
    h, w = shape
    canvas = np.full((h, w), -1, dtype=np.int32)
    order = sorted(range(len(masks)), key=lambda i: -masks[i]["area"])
    for i in order:
        if masks[i]["area"] <= max_area_frac * h * w:
            canvas[masks[i]["segmentation"]] = i
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return canvas[py, px]


@dataclass
class Instances:
    labels: np.ndarray  # (N,) instance id per point, -1 = unassigned
    n_views: np.ndarray  # (N,) number of views in which the point had a mask
    edge_affinity: dict  # instance id -> mean affinity over its internal links


def cluster_instances(
    view_ids: list[tuple[np.ndarray, np.ndarray]], n_points: int,
    min_views: int = 2, min_ratio: float = 0.6, max_group: int = 2000, min_size: int = 5,
) -> Instances:
    """Group points that share masks across views.

    ``view_ids`` holds, per image, (point indices, mask id per point).
    """
    n_views = np.zeros(n_points, dtype=np.int32)
    rows, cols = [], []
    for pts, ids in view_ids:
        ok = ids >= 0
        pts, ids = pts[ok], ids[ok]
        n_views[pts] += 1
        for mid in np.unique(ids):
            group = pts[ids == mid]
            if 2 <= len(group) <= max_group:
                a, b = np.triu_indices(len(group), k=1)
                rows.append(group[a])
                cols.append(group[b])
    if not rows:
        return Instances(np.full(n_points, -1), n_views, {})

    r, c = np.concatenate(rows), np.concatenate(cols)
    co = coo_matrix((np.ones(len(r), dtype=np.int32), (r, c)), shape=(n_points, n_points)).tocsr()
    co = co.tocoo()  # duplicates are summed by tocsr
    ratio = co.data / np.minimum(n_views[co.row], n_views[co.col]).clip(1)
    keep = (co.data >= min_views) & (ratio >= min_ratio)
    link = coo_matrix(
        (np.ones(keep.sum()), (co.row[keep], co.col[keep])), shape=(n_points, n_points)
    )
    _, comp = connected_components(link, directed=False)

    labels = np.full(n_points, -1, dtype=np.int64)
    sizes = np.bincount(comp)
    next_id = 0
    remap = {}
    for k in np.argsort(-sizes):
        if sizes[k] < min_size:
            break
        remap[k] = next_id
        next_id += 1
    for k, new in remap.items():
        labels[comp == k] = new

    affinity = {}
    for new in range(next_id):
        inside = (labels[co.row[keep]] == new) & (labels[co.col[keep]] == new)
        affinity[new] = float(ratio[keep][inside].mean()) if inside.any() else 0.0
    return Instances(labels, n_views, affinity)
