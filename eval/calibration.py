"""Does the precision a reconstruction predicts for itself match its real error?

Photogrammetry reports a precision for every triangulated point from the geometry of its
rays. Remote-sensing validation asks the complementary question: when an estimate is checked
against independent reference data, does the stated uncertainty match the observed error?
This module computes the predicted precision of each sparse point and compares it with the
point-to-plane distance to a laser scan, after the best Sim(3) fit of the camera centres to
the reference poses.

Predicted precision is the standard linearised triangulation covariance with the cameras
held fixed. A ray of length ``r`` seen with focal length ``f`` pixels and image noise
``sigma_px`` has lateral standard deviation ``sigma_px * r / f``; the point's information
matrix is the sum over its rays of ``(f / (sigma_px * r))^2 * (I - d d^T)``. It ignores pose
and calibration uncertainty, so it is a lower bound on the real error.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from registration.sim3 import Sim3


@dataclass
class PointGeometry:
    xyz: np.ndarray  # (N, 3) in reconstruction units
    covariance: np.ndarray  # (N, 3, 3) reconstruction units^2, for sigma_px = 1
    reproj_px: np.ndarray  # (N,) mean reprojection error reported by COLMAP
    track_length: np.ndarray  # (N,)
    max_angle_deg: np.ndarray  # (N,) largest angle between two rays of the track
    mean_range: np.ndarray  # (N,) mean distance to the observing cameras, reconstruction units


def track_geometry(x: np.ndarray, centres: np.ndarray, focals: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Covariance (for ``sigma_px = 1``), largest ray angle in degrees and mean range of one point.

    ``centres`` (m, 3) are the observing camera centres, ``focals`` (m,) their focal lengths in pixels.
    """
    v = x - centres
    r = np.linalg.norm(v, axis=1)
    d = v / r[:, None]
    info = np.einsum("i,ijk->jk", (focals / r) ** 2, np.eye(3)[None] - d[:, :, None] * d[:, None, :])
    max_angle = float(np.degrees(np.arccos(np.clip((d @ d.T).min(), -1.0, 1.0))))
    return np.linalg.pinv(info + 1e-12 * np.eye(3)), max_angle, float(r.mean())


def point_geometry(model_dir: Path) -> PointGeometry:
    """Triangulation geometry of every 3D point of a COLMAP model."""
    rec = pycolmap.Reconstruction(str(model_dir))
    centres = {i: np.asarray(im.projection_center(), dtype=np.float64) for i, im in rec.images.items()}
    focal = {i: float(rec.cameras[im.camera_id].mean_focal_length()) for i, im in rec.images.items()}
    xyz, cov, err, track, ang, rng = [], [], [], [], [], []
    for p in rec.points3D.values():
        ids = [el.image_id for el in p.track.elements]
        x = np.asarray(p.xyz, dtype=np.float64)
        c, a, r = track_geometry(x, np.array([centres[i] for i in ids]), np.array([focal[i] for i in ids]))
        xyz.append(x)
        cov.append(c)
        err.append(p.error)
        track.append(len(ids))
        ang.append(a)
        rng.append(r)
    return PointGeometry(
        np.array(xyz), np.array(cov), np.array(err), np.array(track, dtype=np.int64),
        np.array(ang), np.array(rng),
    )


def scan_normals(scan: np.ndarray, radius: float = 0.06) -> np.ndarray:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return np.asarray(pc.normals)


def actual_error(points_ref: np.ndarray, scan: np.ndarray, normals: np.ndarray, max_nn: float = 0.10):
    """Point-to-plane distance to the scan. Returns (abs error, normal at the match, kept mask).

    Points farther than ``max_nn`` from any scan point are not on a scanned surface (occluded,
    or outside the scan) and are left out rather than counted as errors.
    """
    d, j = cKDTree(scan).query(points_ref)
    n = normals[j]
    e = np.abs(np.einsum("ij,ij->i", points_ref - scan[j], n))
    return e, n, d < max_nn


def predicted_sigma(geo: PointGeometry, transform: Sim3, normals_ref: np.ndarray, sigma_px: float) -> np.ndarray:
    """Predicted standard deviation (metres) along the scan normal at each point."""
    n_rec = normals_ref @ transform.rotation  # normals expressed in the reconstruction frame
    var = np.einsum("ij,ijk,ik->i", n_rec, geo.covariance, n_rec) * sigma_px ** 2
    return transform.scale * np.sqrt(np.maximum(var, 0.0))


def sparsification(error: np.ndarray, ranking: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """RMS error of the points that remain after removing the ``fractions`` ranked worst first."""
    order = np.argsort(-ranking)
    e2 = error[order] ** 2
    n = len(error)
    tail = np.cumsum(e2[::-1])[::-1]  # tail[i] = sum of e2[i:]
    out = []
    for f in fractions:
        k = int(round(f * n))
        out.append(np.sqrt(tail[k] / (n - k)) if k < n else np.nan)
    return np.array(out)


def ause(error: np.ndarray, ranking: np.ndarray, n_steps: int = 20) -> float:
    """Area under the sparsification error curve against the oracle ordering (0 is perfect)."""
    fractions = np.linspace(0, 0.95, n_steps)
    return float(np.nanmean(sparsification(error, ranking, fractions) - sparsification(error, error, fractions)))


def reliability(error: np.ndarray, sigma: np.ndarray, n_bins: int = 8) -> list[dict]:
    """Observed error per quantile bin of the predicted sigma."""
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sigma >= lo) & (sigma <= hi if hi == edges[-1] else sigma < hi)
        if sel.any():
            out.append({
                "n": int(sel.sum()), "predicted_sigma_m": float(sigma[sel].mean()),
                "observed_rms_m": float(np.sqrt((error[sel] ** 2).mean())),
                "observed_median_m": float(np.median(error[sel])),
            })
    return out


def evaluate(geo: PointGeometry, transform: Sim3, scan: np.ndarray, normals: np.ndarray) -> dict:
    """Compare every uncertainty predictor with the observed error, and calibrate the prediction."""
    ref = transform.apply(geo.xyz)
    err, n, kept = actual_error(ref, scan, normals)
    sigma_px = float(np.sqrt((geo.reproj_px ** 2).mean()))
    sigma = predicted_sigma(geo, transform, n, sigma_px)
    e, s = err[kept], sigma[kept]
    predictors = {
        "predicted_sigma": s,
        "reprojection_error": geo.reproj_px[kept],
        "inverse_track_length": 1.0 / geo.track_length[kept],
        "inverse_max_angle": 1.0 / np.maximum(geo.max_angle_deg[kept], 1e-3),
        "range": geo.mean_range[kept] * transform.scale,
    }
    rng = np.random.default_rng(0)
    rank = {k: float(spearmanr(v, e)[0]) for k, v in predictors.items()}
    areas = {k: ause(e, v) for k, v in predictors.items()}
    areas["random"] = float(np.mean([ause(e, rng.random(len(e))) for _ in range(20)]))
    # A single factor that matches the predicted spread to the observed one.
    inflation = float(np.sqrt((e ** 2).mean()) / np.sqrt((s ** 2).mean()))
    calibrated = s * inflation
    return {
        "n_points": int(len(err)), "n_on_scan": int(kept.sum()), "sigma_px": sigma_px,
        "observed_error_m": {"median": float(np.median(e)), "rms": float(np.sqrt((e ** 2).mean())),
                             "p95": float(np.percentile(e, 95))},
        "predicted_sigma_m": {"median": float(np.median(s)), "rms": float(np.sqrt((s ** 2).mean()))},
        "inflation_factor": inflation,
        "coverage_raw": {f"{k}sigma": float((e <= k * s).mean()) for k in (1, 2, 3)},
        "coverage_calibrated": {f"{k}sigma": float((e <= k * calibrated).mean()) for k in (1, 2, 3)},
        "spearman_vs_error": rank,
        "ause_m": areas,
        "reliability": reliability(e, s),
    }
