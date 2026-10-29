"""Accuracy metrics for reconstructions evaluated against reference data."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from eval.colmap_io import centre_from_pose
from registration.sim3 import Sim3, umeyama


@dataclass
class PoseReport:
    n_images: int
    scale: float
    ate_rmse_m: float  # absolute trajectory error of camera centres, metres
    ate_median_m: float
    ate_max_m: float
    rotation_err_median_deg: float
    rotation_err_max_deg: float
    transform: Sim3
    per_image: dict[str, tuple[float, float]]  # name -> (position err m, rotation err deg)

    def to_dict(self) -> dict:
        return {
            "n_images": self.n_images,
            "scale": self.scale,
            "ate_rmse_m": self.ate_rmse_m,
            "ate_median_m": self.ate_median_m,
            "ate_max_m": self.ate_max_m,
            "rotation_err_median_deg": self.rotation_err_median_deg,
            "rotation_err_max_deg": self.rotation_err_max_deg,
        }


def rotation_angle_deg(r: np.ndarray) -> float:
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def pose_errors(est: dict, ref: dict) -> PoseReport:
    """Compare estimated and reference camera poses.

    A Sim(3) is fitted on the camera centres (SfM has an arbitrary frame and
    scale), then position and orientation errors are measured after alignment.
    Poses are matched by image name.
    """
    names = sorted(set(est) & set(ref))
    if len(names) < 3:
        raise ValueError(f"only {len(names)} images in common; need at least 3")

    c_est = np.array([centre_from_pose(*est[n]) for n in names])
    c_ref = np.array([centre_from_pose(*ref[n]) for n in names])
    sim = umeyama(c_est, c_ref, with_scale=True)

    pos_err = np.linalg.norm(sim.apply(c_est) - c_ref, axis=1)

    # Camera-to-world rotation of an estimate after alignment is R_align @ R_cw^T.
    rot_err = []
    for n in names:
        r_est_cw = est[n][0]
        r_ref_cw = ref[n][0]
        r_aligned = r_est_cw @ sim.rotation.T  # world rotation applied on the right
        rot_err.append(rotation_angle_deg(r_aligned @ r_ref_cw.T))
    rot_err = np.array(rot_err)

    return PoseReport(
        n_images=len(names),
        scale=sim.scale,
        ate_rmse_m=float(np.sqrt((pos_err ** 2).mean())),
        ate_median_m=float(np.median(pos_err)),
        ate_max_m=float(pos_err.max()),
        rotation_err_median_deg=float(np.median(rot_err)),
        rotation_err_max_deg=float(rot_err.max()),
        transform=sim,
        per_image={n: (float(p), float(r)) for n, p, r in zip(names, pos_err, rot_err)},
    )


def accuracy_completeness(
    est_points: np.ndarray, ref_points: np.ndarray, thresholds=(0.02, 0.05, 0.10)
) -> dict:
    """Point-cloud accuracy / completeness against a reference scan.

    accuracy     : distance from each estimated point to the nearest reference point
    completeness : distance from each reference point to the nearest estimated point
    Reported as mean/median distance plus the fraction within each threshold
    (precision, recall) and their harmonic mean (F-score). Distances in the
    unit of the inputs (metres).
    """
    d_acc = cKDTree(ref_points).query(est_points)[0]
    d_comp = cKDTree(est_points).query(ref_points)[0]
    out = {
        "n_est": int(len(est_points)),
        "n_ref": int(len(ref_points)),
        "accuracy_mean_m": float(d_acc.mean()),
        "accuracy_median_m": float(np.median(d_acc)),
        "completeness_mean_m": float(d_comp.mean()),
        "completeness_median_m": float(np.median(d_comp)),
        "at_threshold": {},
    }
    for tau in thresholds:
        precision = float((d_acc <= tau).mean())
        recall = float((d_comp <= tau).mean())
        f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out["at_threshold"][f"{tau:g}"] = {
            "precision": precision, "recall": recall, "f_score": f,
        }
    return out


def reprojection_stats(error_px: np.ndarray, track_len: np.ndarray) -> dict:
    """Summary statistics of per-point reprojection error and track length."""
    return {
        "n_points": int(len(error_px)),
        "reproj_mean_px": float(error_px.mean()),
        "reproj_median_px": float(np.median(error_px)),
        "reproj_p95_px": float(np.percentile(error_px, 95)),
        "reproj_max_px": float(error_px.max()),
        "track_len_mean": float(track_len.mean()),
        "track_len_min": int(track_len.min()),
        "frac_tracks_ge3": float((track_len >= 3).mean()),
    }
