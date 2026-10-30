"""Structural scene understanding for rooms: planes and per-point labels.

Pipeline: dominant orthogonal axes (Manhattan frame) -> vertical axis from the
cameras' up direction -> floor, ceiling and wall planes as peaks of the point
distribution along each axis -> per-point label with a confidence.

Everything is derived from the points and camera orientations. Labels and
confidences are inferences: a plane that has little support is reported as such,
and the ceiling is only reported when it is actually observed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from registration.manhattan import estimate_normals, manhattan_axes
from registration.pipeline import normalise

OBJECT = "object"


@dataclass
class Plane:
    role: str  # "floor" | "ceiling" | "wall"
    axis_index: int  # which room axis the plane is perpendicular to
    offset: float  # coordinate along that axis
    support: int  # number of points within the plane tolerance
    sigma: float  # robust spread of those points about the plane
    side: int = 0  # -1 low end, +1 high end of the axis

    @property
    def name(self) -> str:
        if self.role in ("floor", "ceiling"):
            return self.role
        return f"wall_{self.axis_index}_{'lo' if self.side < 0 else 'hi'}"


@dataclass
class RoomFrame:
    axes: np.ndarray  # (3, 3), columns are unit axes; column `up` points up
    up: int


def room_frame(points: np.ndarray, cam_up: np.ndarray, normal_voxel: float = 0.03) -> RoomFrame:
    """Room axes with the vertical one chosen from the cameras' mean up vector."""
    pts_n, _, _ = normalise(points)  # normalisation only changes units, not directions
    axes = manhattan_axes(estimate_normals(pts_n, normal_voxel))
    cam_up = cam_up / np.linalg.norm(cam_up)
    up = int(np.argmax(np.abs(axes.T @ cam_up)))
    if axes[:, up] @ cam_up < 0:
        axes = axes.copy()
        axes[:, up] *= -1
        axes[:, (up + 1) % 3] *= -1  # keep right-handed
    return RoomFrame(axes, up)


def camera_up(rotations_cw: list[np.ndarray]) -> np.ndarray:
    """Mean world-space up vector of cameras (COLMAP camera y points down)."""
    ups = np.array([-r.T @ np.array([0.0, 1.0, 0.0]) for r in rotations_cw])
    mean = ups.mean(axis=0)
    return mean / np.linalg.norm(mean)


def _peaks(u: np.ndarray, lo: float, hi: float, bins: int = 200, min_frac: float = 0.08) -> list[tuple[float, int]]:
    """Peaks of the coordinate histogram over [lo, hi], lowest first."""
    # Pad with empty bins: planes at the very ends of the range must still be peaks.
    pad = 5 * (hi - lo) / bins
    hist, edges = np.histogram(u, bins=bins, range=(lo - pad, hi + pad))
    smooth = gaussian_filter1d(hist.astype(float), 1.0)
    idx, _ = find_peaks(smooth, height=min_frac * smooth.max(), prominence=0.05 * smooth.max())
    centres = (edges[:-1] + edges[1:]) / 2
    return [(float(centres[i]), int(hist[i])) for i in idx]


def detect_planes(
    points: np.ndarray, frame: RoomFrame, cam_centres: np.ndarray | None = None,
    min_separation: float = 0.3,
) -> list[Plane]:
    """Floor, ceiling and walls as histogram peaks along each room axis.

    With ``cam_centres`` (the capture positions, inside the room) the roles follow
    from geometry: the low-end plane is the lowest peak *below* every camera, the
    high-end plane the highest peak *above* every camera. A side without such a peak
    gets no plane, so a weakly observed floor or ceiling is reported as absent
    instead of being replaced by a different surface. Without cameras the outermost
    two peaks are used (less reliable). Extents use 0.5/99.5 percentiles because
    sparse reconstructions contain far outliers.
    """
    coords = points @ frame.axes
    cams = None if cam_centres is None else cam_centres @ frame.axes
    planes = []
    for axis in range(3):
        u = coords[:, axis]
        lo, hi = np.percentile(u, [0.5, 99.5])
        extent = hi - lo
        peaks = _peaks(u, lo, hi)
        if not peaks:
            continue
        if cams is not None:
            below = [pk for pk in peaks if pk[0] < cams[:, axis].min()]
            above = [pk for pk in peaks if pk[0] > cams[:, axis].max()]
            chosen = ([(below[0], -1)] if below else []) + ([(above[-1], +1)] if above else [])
        else:
            chosen = [(peaks[0], -1)]
            if peaks[-1][0] - peaks[0][0] >= min_separation * extent:
                chosen.append((peaks[-1], +1))
        for (pos, _), side in chosen:
            near = np.abs(u - pos) < 0.02 * extent
            if near.sum() > 2:  # refine the bin centre to the median of the supporting points
                pos = float(np.median(u[near]))
                near = np.abs(u - pos) < 0.02 * extent
            dist = u[near] - pos
            sigma = float(1.4826 * np.median(np.abs(dist))) if near.sum() > 2 else 0.01 * extent
            sigma = max(sigma, 0.002 * extent)
            if axis == frame.up:
                role = "floor" if side < 0 else "ceiling"
            else:
                role = "wall"
            planes.append(Plane(role, axis, pos, int(near.sum()), sigma, side))
    return planes


def label_points(
    points: np.ndarray, frame: RoomFrame, planes: list[Plane], min_sigma: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Label each point with its nearest plane (within 3 sigma) or ``object``.

    Returns (labels (N,) of str, confidence (N,) in [0, 1]). For plane labels the
    confidence is a Gaussian falloff of the distance in units of the plane's sigma;
    for ``object`` it grows with the distance to the nearest plane.
    """
    coords = points @ frame.axes
    labels = np.full(len(points), OBJECT, dtype=object)
    conf = np.zeros(len(points))
    if not planes:
        return labels, np.ones(len(points))
    # ``min_sigma`` floors the tolerance, e.g. when labelling noisy points with
    # planes that were fitted on a much more precise cloud.
    dist = np.stack(
        [np.abs(coords[:, p.axis_index] - p.offset) / max(p.sigma, min_sigma) for p in planes], axis=1
    )
    nearest = dist.argmin(axis=1)
    d_min = dist[np.arange(len(points)), nearest]
    on_plane = d_min < 3.0
    names = np.array([p.name for p in planes], dtype=object)
    labels[on_plane] = names[nearest[on_plane]]
    conf[on_plane] = np.exp(-0.5 * d_min[on_plane] ** 2)
    conf[~on_plane] = np.minimum(1.0, d_min[~on_plane] / 6.0)
    return labels, conf
