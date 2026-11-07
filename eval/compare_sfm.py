"""Compare two SfM engines on the same matches: COLMAP incremental and COLMAP global mapping.

Both engines read a copy of one COLMAP database (features and verified matches computed once, by
the controller), so only the mapping step differs. Each run keeps the model with the most registered
images, as the controller does, and is scored against the ETH3D reference poses and laser scan
with the same metrics as ``eval.run_eval``. The random seed is varied, because both engines are
randomised.

Usage (from the repo root):
    python -m eval.compare_sfm --database outputs/office_agent/opencv/database.db \
        --images data/eth3d/office/images/dslr_images --scene data/eth3d/office \
        --out outputs/sfm_engines/office --seeds 3
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

from eval.colmap_io import (
    load_reference_scan,
    read_model_points,
    read_model_poses,
    read_reference_poses,
)
from eval.metrics import accuracy_completeness, pose_errors, reprojection_stats
from sfm.view_graph import drop_inconsistent_pairs

ENGINES = ("incremental", "global", "global_filtered")


def final_bundle_adjustment(rec: pycolmap.Reconstruction, refine_principal_point: bool) -> None:
    """One more joint refinement of points, poses and intrinsics, as the incremental mapper ends with."""
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = True
    options.refine_extra_params = True
    options.refine_principal_point = refine_principal_point
    pycolmap.bundle_adjustment(rec, options)


def run_engine(engine: str, database: Path, images: Path, out: Path, seed: int,
               min_matches: int | None = None, refine_principal_point: bool = False) -> tuple[Path | None, float, int]:
    """Run one engine on ``database``; return (best model folder, seconds, number of models).

    The database is copied first so neither engine can change the other's input.
    ``min_matches`` overrides COLMAP's minimum number of verified matches for an image pair
    (15 by default, for all engines). ``global_filtered`` first drops the pairs whose rotation
    loops do not close (``sfm.view_graph``) and finishes with one more bundle adjustment; its time
    includes both. ``refine_principal_point`` lets every engine refine the principal point.
    """
    out.mkdir(parents=True, exist_ok=True)
    db = out / "database.db"
    shutil.copy(database, db)
    sparse = out / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    start = time.perf_counter()
    if engine == "incremental":
        options = pycolmap.IncrementalPipelineOptions()
        options.random_seed = seed
        options.ba_refine_principal_point = refine_principal_point
        if min_matches is not None:
            options.min_num_matches = min_matches
        maps = pycolmap.incremental_mapping(db, images, sparse, options)
    elif engine in ("global", "global_filtered"):
        if engine == "global_filtered":
            drop_inconsistent_pairs(db)
        options = pycolmap.GlobalPipelineOptions()
        options.mapper.bundle_adjustment.refine_principal_point = refine_principal_point
        options.random_seed = seed
        options.mapper.random_seed = seed
        if min_matches is not None:
            options.min_num_matches = min_matches
        maps = pycolmap.global_mapping(db, images, sparse, options)
        if engine == "global_filtered":
            for rec in maps.values():
                final_bundle_adjustment(rec, refine_principal_point)
    else:
        raise ValueError(f"unknown engine {engine!r}")
    seconds = time.perf_counter() - start
    if not maps:
        return None, seconds, 0
    best_idx = max(maps, key=lambda i: maps[i].num_reg_images())
    best = out / "best"
    if best.exists():
        shutil.rmtree(best)
    best.mkdir()
    maps[best_idx].write(best)
    return best, seconds, len(maps)


def score(model: Path, ref_poses: dict, scan: np.ndarray) -> dict:
    """Pose, point and reprojection metrics of one model, as ``eval.run_eval`` computes them."""
    pose = pose_errors(read_model_poses(model), ref_poses)
    xyz, err, track = read_model_points(model)
    points = accuracy_completeness(pose.transform.apply(xyz), scan)
    return {
        "poses": pose.to_dict(),
        "accuracy_median_m": points["accuracy_median_m"],
        "precision_2cm": points["at_threshold"]["0.02"]["precision"],
        "reprojection": reprojection_stats(err, track),
    }


def summarise(runs: list[dict]) -> dict:
    """Per engine: mean, standard deviation and range over seeds of the headline numbers."""
    fields = {
        "seconds": lambda r: r["seconds"],
        "n_registered": lambda r: r["n_registered"],
        "n_points": lambda r: r["n_points"],
        "camera_rmse_cm": lambda r: r["poses"]["ate_rmse_m"] * 100,
        "rotation_median_deg": lambda r: r["poses"]["rotation_err_median_deg"],
        "point_median_mm": lambda r: r["accuracy_median_m"] * 1000,
        "within_2cm": lambda r: r["precision_2cm"],
        "reproj_mean_px": lambda r: r["reprojection"]["reproj_mean_px"],
    }
    out = {}
    for engine in sorted({r["engine"] for r in runs}):
        mine = [r for r in runs if r["engine"] == engine and r.get("n_registered")]
        out[engine] = {"n_runs": len([r for r in runs if r["engine"] == engine]), "n_scored": len(mine)}
        for name, get in fields.items():
            v = np.array([get(r) for r in mine], dtype=float)
            if len(v):
                out[engine][name] = {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True, help="COLMAP database with matches")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True, help="ETH3D scene folder")
    parser.add_argument("--out", type=Path, required=True, help="folder for models; <out>/results.json is written")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--refine-principal-point", action="store_true", help="every engine refines the principal point")
    parser.add_argument("--min-matches", type=int, help="minimum verified matches for an image pair (COLMAP default 15)")
    parser.add_argument("--voxel", type=float, default=0.02, help="reference scan voxel size (m)")
    args = parser.parse_args()

    ref_poses = read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt")
    scan = load_reference_scan(args.scene / "scan_clean", voxel=args.voxel)
    n_images = len([p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    runs = []
    for engine in args.engines.split(","):
        for seed in range(1, args.seeds + 1):
            model, seconds, n_models = run_engine(engine, args.database, args.images, args.out / f"{engine}_seed{seed}", seed, args.min_matches, args.refine_principal_point)
            run = {"engine": engine, "seed": seed, "seconds": round(seconds, 2), "n_models": n_models,
                   "n_images": n_images, "min_matches": args.min_matches, "refine_principal_point": args.refine_principal_point, "model": str(model) if model else None}
            if model is not None:
                rec = pycolmap.Reconstruction(str(model))
                run["n_registered"] = rec.num_reg_images()
                run["n_points"] = rec.num_points3D()
                run.update(score(model, ref_poses, scan))
            runs.append(run)
            print(json.dumps({k: run[k] for k in ("engine", "seed", "seconds", "n_registered", "n_points")} |
                             ({"ate_rmse_cm": round(run["poses"]["ate_rmse_m"] * 100, 2)} if model else {}), default=str),
                  flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarise(runs)
    (args.out / "results.json").write_text(json.dumps({"scene": str(args.scene), "summary": summary, "runs": runs}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
