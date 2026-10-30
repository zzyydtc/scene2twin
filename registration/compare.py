"""Compare registration strategies on a reconstruction vs. a reference scan.

Usage (from the repo root):
    python -m registration.compare --model outputs/office_fisheye/best \
        --scene data/eth3d/office --out outputs/office_fisheye/registration.json

Strategies (all map the SfM points into the scan frame, scale unknown):
  oracle_centres : Sim(3) on camera centres vs. reference poses. Uses ground-truth
                   poses, so it is a reference point, not a deployable method.
  open3d_default : Open3D FPFH + RANSAC (with scaling) + ICP (with scaling).
  ours           : registration.pipeline.register (own RANSAC and ICP).
  ours_manhattan : registration.manhattan.register_manhattan (room-axes coarse stage).
  ours_icp_only  : own ICP started from the oracle_centres transform.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from eval.colmap_io import (
    centre_from_pose, load_reference_scan, read_model_points, read_model_poses,
    read_reference_poses,
)
from eval.metrics import accuracy_completeness
from registration.manhattan import register_manhattan
from registration.pipeline import (
    compose, fpfh, icp_sim3, inverse_similarity, normalise, register, similarity,
)
from registration.sim3 import Sim3, umeyama


def score(points: np.ndarray, scan: np.ndarray, transform: Sim3,
          centres: tuple[np.ndarray, np.ndarray]) -> dict:
    """Point distance to the scan plus camera-centre error under the transform.

    The point distance alone is one-sided and can be gamed by shrinking the
    reconstruction onto the scan, so camera centres (est, ref) are also compared.
    """
    a = accuracy_completeness(transform.apply(points), scan, thresholds=(0.02, 0.05))
    c_est, c_ref = centres
    cam_err = np.linalg.norm(transform.apply(c_est) - c_ref, axis=1)
    return {
        "scale": transform.scale,
        "camera_rmse_m": float(np.sqrt((cam_err ** 2).mean())),
        "median_m": a["accuracy_median_m"],
        "mean_m": a["accuracy_mean_m"],
        "within_2cm": a["at_threshold"]["0.02"]["precision"],
        "within_5cm": a["at_threshold"]["0.05"]["precision"],
    }


def camera_centres(est_poses: dict, ref_poses: dict) -> tuple[np.ndarray, np.ndarray]:
    names = sorted(set(est_poses) & set(ref_poses))
    c_est = np.array([centre_from_pose(*est_poses[n]) for n in names])
    c_ref = np.array([centre_from_pose(*ref_poses[n]) for n in names])
    return c_est, c_ref


def open3d_default(src: np.ndarray, dst: np.ndarray, voxel: float, seed: int) -> Sim3:
    """Library baseline on identically normalised inputs and the same voxel size."""
    o3d.utility.random.seed(seed)
    src_n, c_s, r_s = normalise(src)
    dst_n, c_d, r_d = normalise(dst)
    clouds, feats = [], []
    for pts in (src_n, dst_n):
        p, f = fpfh(pts, voxel)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        ft = o3d.pipelines.registration.Feature()
        ft.data = f.T.copy()
        clouds.append(pc)
        feats.append(ft)
    reg = o3d.pipelines.registration
    coarse = reg.registration_ransac_based_on_feature_matching(
        clouds[0], clouds[1], feats[0], feats[1], True, 2.0 * voxel,
        reg.TransformationEstimationPointToPoint(True), 3,
        [reg.CorrespondenceCheckerBasedOnDistance(2.0 * voxel)],
        reg.RANSACConvergenceCriteria(100000, 0.999),
    )
    s_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_n))
    d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst_n))
    fine = reg.registration_icp(
        s_pc, d_pc, 1.5 * voxel, coarse.transformation,
        reg.TransformationEstimationPointToPoint(True),
    )
    m = fine.transformation
    lin = m[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(lin))))
    fine_n = Sim3(scale, lin / scale, m[:3, 3])
    return compose(inverse_similarity(r_d, c_d), compose(fine_n, similarity(r_s, c_s)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.06, help="normalised units")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    scan = load_reference_scan(args.scene / "scan_clean", voxel=0.02)
    pts, _, _ = read_model_points(args.model)
    est_poses = read_model_poses(args.model)
    ref_poses = read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt")

    results: dict = {}
    centres = camera_centres(est_poses, ref_poses)
    oracle = umeyama(*centres)
    results["oracle_centres"] = score(pts, scan, oracle, centres)

    t0 = time.perf_counter()
    refined = icp_sim3(pts, scan, oracle, max_dist=0.15)
    results["ours_icp_only"] = {**score(pts, scan, refined, centres), "seconds": time.perf_counter() - t0}

    for name, fn in (
        ("open3d_default", lambda s: open3d_default(pts, scan, args.voxel, s)),
        ("ours", lambda s: register(pts, scan, voxel=args.voxel, seed=s).transform),
        ("ours_manhattan", lambda s: register_manhattan(pts, scan, seed=s).transform),
    ):
        runs = []
        for seed in range(args.seeds):
            t0 = time.perf_counter()
            try:
                runs.append({**score(pts, scan, fn(seed), centres), "seconds": time.perf_counter() - t0})
            except Exception as exc:  # report a failed run instead of hiding it
                runs.append({"error": str(exc)})
        results[name] = runs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for name, val in results.items():
        if isinstance(val, dict):
            print(f"{name:15s} cam {val['camera_rmse_m']*100:7.2f} cm  median {val['median_m']*100:6.2f} cm  <=2cm {val['within_2cm']:.2f}  scale {val['scale']:.4f}")
        else:
            for i, r in enumerate(val):
                if "error" in r:
                    print(f"{name:15s} seed {i}: ERROR {r['error']}")
                else:
                    print(f"{name:15s} seed {i}: cam {r['camera_rmse_m']*100:7.2f} cm  median {r['median_m']*100:6.2f} cm  <=2cm {r['within_2cm']:.2f}  scale {r['scale']:.4f}  {r['seconds']:.1f}s")


if __name__ == "__main__":
    main()
