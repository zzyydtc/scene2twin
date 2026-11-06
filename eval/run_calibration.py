"""Predicted precision versus observed error of a COLMAP model, against an ETH3D scan.

Usage (from the repo root):
    python -m eval.run_calibration --model outputs/office_agent/opencv/best \
        --scene data/eth3d/office --out outputs/calibration/office.json --figure docs/figures/office_calibration.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.calibration import actual_error, evaluate, point_geometry, predicted_sigma, scan_normals, sparsification
from eval.colmap_io import load_reference_scan, read_model_poses, read_reference_poses
from eval.metrics import pose_errors


def draw(report: dict, geo_data: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rel = report["reliability"]
    x = np.array([b["predicted_sigma_m"] for b in rel]) * 1000
    y = np.array([b["observed_rms_m"] for b in rel]) * 1000
    lim = max(x.max(), y.max()) * 1.1
    ax[0].plot([0, lim], [0, lim], color="0.6", lw=1, label="perfectly calibrated")
    ax[0].plot(x, y, "o-", color="#1f6f8b", label="observed RMS in each bin")
    ax[0].set_xlabel("predicted sigma along the scan normal (mm)")
    ax[0].set_ylabel("observed point-to-plane error (mm)")
    ax[0].set_title("Reliability")
    ax[0].legend(frameon=False, fontsize=8)

    e = geo_data["error"]
    fractions = np.linspace(0, 0.95, 20)
    ax[1].plot(fractions, sparsification(e, e, fractions) * 1000, color="0.4", ls="--", label="oracle")
    for key, colour, label in (
        ("predicted_sigma", "#1f6f8b", "predicted sigma"),
        ("reprojection_error", "#c8553d", "reprojection error"),
        ("inverse_max_angle", "#6a994e", "1 / ray angle"),
    ):
        ax[1].plot(fractions, sparsification(e, geo_data["rank"][key], fractions) * 1000, color=colour, label=label)
    ax[1].set_xlabel("fraction of points removed (worst ranked first)")
    ax[1].set_ylabel("RMS error of the rest (mm)")
    ax[1].set_title("Sparsification")
    ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    pose = pose_errors(read_model_poses(args.model), read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt"))
    geo = point_geometry(args.model)
    scan = load_reference_scan(args.scene / "scan_clean", voxel=0.02)
    normals = scan_normals(scan)
    report = evaluate(geo, pose.transform, scan, normals)
    report["pose_fit"] = {"scale": pose.scale, "ate_rmse_m": pose.ate_rmse_m}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    if args.figure:
        ref = pose.transform.apply(geo.xyz)
        err, n, kept = actual_error(ref, scan, normals)
        sigma = predicted_sigma(geo, pose.transform, n, report["sigma_px"])
        rank = {
            "predicted_sigma": sigma[kept],
            "reprojection_error": geo.reproj_px[kept],
            "inverse_max_angle": 1.0 / np.maximum(geo.max_angle_deg[kept], 1e-3),
        }
        draw(report, {"error": err[kept], "rank": rank}, args.figure, f"{args.scene.name}: {report['n_on_scan']} points on the scan")

    print(json.dumps({k: v for k, v in report.items() if k != "reliability"}, indent=2))


if __name__ == "__main__":
    main()
