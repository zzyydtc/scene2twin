"""Start COLMAP from VGGT's cameras: triangulate the database matches, then bundle-adjust.

VGGT's cameras are good but not accurate (see docs/day10-vggt.md). This module gives them to COLMAP as
the starting poses and the starting focal length, and lets the usual machinery finish the job:
triangulate the verified matches of a database with those poses, run bundle adjustment, and repeat
with tighter thresholds. No mapper chooses an initial pair or registers images one by one; the
poses come from the network.

Usage (from the repo root; run ``feedforward.vggt_run`` first):
    python -m feedforward.refine --vggt outputs/vggt/office \
        --database outputs/office_agent/opencv/database.db \
        --images data/eth3d/office/images/dslr_images --scene data/eth3d/office --out outputs/vggt/office_refined
"""
import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from eval.colmap_io import load_reference_scan, read_reference_poses
from eval.compare_sfm import final_bundle_adjustment, score

# (maximum reprojection error in px, maximum triangulation angle error in degrees) per round: loose
# first, because the starting poses are centimetres off, then COLMAP's usual values until nothing changes.
ROUNDS = [(24.0, 8.0), (12.0, 6.0), (8.0, 4.0)] + [(4.0, 2.0)] * 10


def initial_model_text(names: list[str], extrinsic: np.ndarray, image_ids: dict, camera_id: int,
                       size: tuple[int, int], focal: tuple[float, float]) -> tuple[str, str, str]:
    """COLMAP text-format (cameras, images, points3D) for VGGT's poses, without any points.

    ``extrinsic`` is [S, 3, 4] camera-from-world in the order of ``names``; ``image_ids`` maps a file
    name to the image id in the database. The camera is OPENCV with VGGT's focal length, the principal
    point at the image centre and no distortion (VGGT assumes none).
    """
    width, height = size
    cameras = f"{camera_id} OPENCV {width} {height} {focal[0]} {focal[1]} {width / 2} {height / 2} 0 0 0 0\n"
    rows = []
    for name, ext in zip(names, extrinsic):
        qx, qy, qz, qw = Rotation.from_matrix(ext[:, :3]).as_quat()  # scipy order is xyzw, COLMAP's is wxyz
        t = ext[:, 3]
        rows += [f"{image_ids[name]} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {camera_id} {name}", ""]
    return cameras, "\n".join(rows) + "\n", ""


def triangulation_options(max_reproj_px: float, max_angle_deg: float) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.triangulation.complete_max_reproj_error = max_reproj_px
    options.triangulation.merge_max_reproj_error = max_reproj_px
    options.triangulation.create_max_angle_error = max_angle_deg
    options.triangulation.continue_max_angle_error = max_angle_deg
    return options


def refine(vggt_dir: Path, database: Path, images: Path, out: Path, refine_principal_point: bool = False) -> tuple[Path, float]:
    """Run the rounds above; write the model to ``out/best`` and return it with the seconds taken."""
    results = json.loads((vggt_dir / "results.json").read_text())
    extrinsic = np.load(vggt_dir / "predictions.npz")["extrinsic"]
    names = sorted(p.name for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database)
    rows = list(db.execute("select image_id, name, camera_id from images"))
    db.close()
    camera_ids = {c for _, _, c in rows}
    if len(camera_ids) != 1:
        raise ValueError("the database must use one shared camera")
    # The photo size is in the database's camera; the focal comes from VGGT.
    cam = pycolmap.Database.open(str(database))
    camera = cam.read_all_cameras()[0]
    cam.close()
    focal_px = results["focal_px"]
    focal = (focal_px.get("fx", focal_px.get("vggt_fx")), focal_px.get("fy", focal_px.get("vggt_fy")))
    texts = initial_model_text(names, extrinsic, {n: i for i, n, _ in rows}, camera_ids.pop(), (camera.width, camera.height), focal)
    text_dir = out / "text"
    shutil.rmtree(text_dir, ignore_errors=True)
    text_dir.mkdir()
    for fname, body in zip(("cameras.txt", "images.txt", "points3D.txt"), texts):
        (text_dir / fname).write_text(body)

    work_db = out / "database.db"
    shutil.copy(database, work_db)
    rec = pycolmap.Reconstruction(str(text_dir))
    start = time.perf_counter()
    for px, angle in ROUNDS:
        rec = pycolmap.triangulate_points(rec, work_db, images, out / "triangulated", options=triangulation_options(px, angle))
        final_bundle_adjustment(rec, refine_principal_point)
    seconds = time.perf_counter() - start
    best = out / "best"
    shutil.rmtree(best, ignore_errors=True)
    best.mkdir()
    rec.write(best)
    return best, seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt", type=Path, required=True, help="folder written by feedforward.vggt_run")
    parser.add_argument("--database", type=Path, required=True, help="COLMAP database with verified matches")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True, help="ETH3D scene folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refine-principal-point", action="store_true")
    args = parser.parse_args()

    model, seconds = refine(args.vggt, args.database, args.images, args.out, args.refine_principal_point)
    ref = read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt")
    result = score(model, ref, load_reference_scan(args.scene / "scan_clean", voxel=0.02))
    rec = pycolmap.Reconstruction(str(model))
    result.update(seconds=round(seconds, 1), n_registered=rec.num_reg_images(), n_points=rec.num_points3D(), refine_principal_point=args.refine_principal_point)
    (args.out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"camera_rmse_cm": round(result["poses"]["ate_rmse_m"] * 100, 2), "rotation_median_deg": round(result["poses"]["rotation_err_median_deg"], 3),
                      "point_median_mm": round(result["accuracy_median_m"] * 1000, 1), "within_2cm": round(result["precision_2cm"], 3),
                      "reproj_px": round(result["reprojection"]["reproj_mean_px"], 2), "n_points": result["n_points"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
