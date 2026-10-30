"""Scene graph from labelled room points: planes, objects and relations.

Every node records how it was obtained: ``measured`` nodes are backed by points
that sit on the plane or form the object; ``not_observed`` marks structure the
reconstruction gives no evidence for (for example a ceiling with no points).
Lengths are in the units of the input points unless ``scale`` converts them to
metres, which should come from a tape-measured reference distance.
"""
import numpy as np
import open3d as o3d

from semantics.structure import OBJECT, Plane, RoomFrame


def scale_from_reference(p1: np.ndarray, p2: np.ndarray, length_m: float) -> float:
    """Metres per reconstruction unit from two points a known distance apart."""
    d = float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))
    if d <= 0 or length_m <= 0:
        raise ValueError("points must be distinct and the length positive")
    return length_m / d


def cluster_objects(
    points: np.ndarray, labels: np.ndarray, frame: RoomFrame, extent: float,
    eps_frac: float = 0.03, min_points: int = 8,
) -> list[np.ndarray]:
    """Index arrays of DBSCAN clusters among the ``object`` points."""
    idx = np.flatnonzero(labels == OBJECT)
    if len(idx) < min_points:
        return []
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx]))
    cl = np.asarray(pc.cluster_dbscan(eps=eps_frac * extent, min_points=min_points))
    return [idx[cl == k] for k in range(cl.max() + 1)] if cl.size and cl.max() >= 0 else []


def build_scene_graph(
    points: np.ndarray, labels: np.ndarray, conf: np.ndarray, frame: RoomFrame,
    planes: list[Plane], scale: float = 1.0, unit: str = "units",
) -> dict:
    coords = points @ frame.axes
    # 0.5 / 99.5 percentiles: sparse reconstructions contain far outliers.
    ext = np.percentile(coords, 99.5, axis=0) - np.percentile(coords, 0.5, axis=0)
    nodes, edges = [], []

    by_role = {p.name: p for p in planes}
    for p in planes:
        sel = labels == p.name
        nodes.append({
            "id": p.name, "type": "plane", "role": p.role, "status": "measured",
            "axis": p.axis_index, "offset": float(p.offset * scale),
            "support_points": int(sel.sum()),
            "mean_confidence": float(conf[sel].mean()) if sel.any() else 0.0,
        })
    if "ceiling" not in by_role:
        nodes.append({"id": "ceiling", "type": "plane", "role": "ceiling", "status": "not_observed"})

    measured = [n for n in nodes if n["type"] == "plane" and n["status"] == "measured"]
    for a in range(len(measured)):
        for b in range(a + 1, len(measured)):
            if measured[a]["axis"] != measured[b]["axis"]:
                edges.append({"from": measured[a]["id"], "to": measured[b]["id"], "relation": "perpendicular"})

    floor = by_role.get("floor")
    up = frame.up
    for k, idx in enumerate(cluster_objects(points, labels, frame, float(ext.max()))):
        c = coords[idx]
        lo, hi = c.min(axis=0), c.max(axis=0)
        oid = f"object_{k}"
        node = {
            "id": oid, "type": "object", "status": "measured", "points": int(len(idx)),
            "size": [float(v * scale) for v in (hi - lo)],
            "mean_confidence": float(conf[idx].mean()),
        }
        if floor is not None:
            node["height_above_floor"] = float((lo[up] - floor.offset) * scale)
            if abs(lo[up] - floor.offset) < 3 * floor.sigma + 0.03 * ext[up]:
                edges.append({"from": oid, "to": "floor", "relation": "on"})
        for p in planes:
            if p.role == "wall":
                gap = min(abs(lo[p.axis_index] - p.offset), abs(hi[p.axis_index] - p.offset))
                if gap < 0.05 * ext[p.axis_index]:
                    edges.append({"from": oid, "to": p.name, "relation": "against"})
        nodes.append(node)

    return {
        "unit": unit,
        "room_extent": [float(v * scale) for v in ext],
        "up_axis": int(up),
        "label_counts": {str(k): int((labels == k).sum()) for k in sorted(set(labels))},
        "nodes": nodes,
        "edges": edges,
    }
