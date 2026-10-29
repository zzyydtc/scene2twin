"""Evaluate a COLMAP model against ETH3D reference data.

Usage (from the repo root):
    python -m eval.run_eval --model outputs/office/sparse/0 \
        --scene data/eth3d/office --out outputs/office/eval.json
"""
import argparse
import json
from pathlib import Path

from eval.colmap_io import (
    load_reference_scan,
    read_model_points,
    read_model_poses,
    read_reference_poses,
)
from eval.metrics import accuracy_completeness, pose_errors, reprojection_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="COLMAP model folder")
    parser.add_argument("--scene", type=Path, required=True, help="ETH3D scene folder")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--voxel", type=float, default=0.02, help="reference scan voxel size (m)")
    args = parser.parse_args()

    ref_poses = read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt")
    est_poses = read_model_poses(args.model)
    pose = pose_errors(est_poses, ref_poses)

    xyz, err, track = read_model_points(args.model)
    xyz_ref_frame = pose.transform.apply(xyz)  # bring points into the reference frame
    ref_cloud = load_reference_scan(args.scene / "scan_clean", voxel=args.voxel)

    report = {
        "poses": pose.to_dict(),
        "points_vs_scan": accuracy_completeness(xyz_ref_frame, ref_cloud),
        "reprojection": reprojection_stats(err, track),
        "note": (
            "Sparse model: accuracy is meaningful; completeness against a dense "
            "scan is expected to be low until a dense reconstruction is evaluated."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
