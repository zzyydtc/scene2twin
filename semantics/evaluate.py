"""Evaluate structural semantics on an ETH3D scene against the laser scan.

Usage (from the repo root):
    python -m semantics.evaluate --model outputs/office_fisheye/best \
        --scene data/eth3d/office --out outputs/office_fisheye/semantics.json

Protocol (mimics a tape-measured demo):
  1. Run the structure pipeline on the SfM points and, separately, on the scan.
  2. One reference dimension taken from the scan plays the role of the tape
     measurement and sets the metric scale of the reconstruction.
  3. The remaining room dimensions are compared with the scan's.
  4. Point labels are compared with labels obtained from the scan's own planes. The
     SfM points are moved into the scan frame with a Sim(3) fitted on camera centres
     against the reference poses; that uses ground truth, for evaluation only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.colmap_io import (
    centre_from_pose, load_reference_scan, read_model_points, read_model_poses, read_reference_poses,
)
from registration.compare import camera_centres
from registration.sim3 import umeyama
from semantics.scene_graph import build_scene_graph
from semantics.structure import OBJECT, camera_up, detect_planes, label_points, room_frame


# Plane tolerance for judging SfM points against the scan: 3 sigma = 3 cm, about twice
# the point error measured on this scene (median 1.35 cm).
EVAL_TOLERANCE_M = 0.03


def analyse(points: np.ndarray, poses: dict):
    frame = room_frame(points, camera_up([r for r, _ in poses.values()]))
    centres = np.array([centre_from_pose(r, t) for r, t in poses.values()])
    planes = detect_planes(points, frame, centres)
    labels, conf = label_points(points, frame, planes)
    return frame, planes, labels, conf


def dimensions(planes, frame) -> dict:
    """Plane-to-plane distances per axis; the vertical one only if both ends exist."""
    out = {}
    for axis in range(3):
        offs = sorted(p.offset for p in planes if p.axis_index == axis)
        if len(offs) >= 2:
            out["height" if axis == frame.up else f"horizontal_{axis}"] = offs[-1] - offs[0]
    return out


def coarse(labels: np.ndarray) -> np.ndarray:
    return np.array(["wall" if str(l).startswith("wall") else str(l) for l in labels], dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    est_poses = read_model_poses(args.model)
    ref_poses = read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt")
    pts, _, _ = read_model_points(args.model)
    scan = load_reference_scan(args.scene / "scan_clean", voxel=0.02)

    f_sfm, planes_sfm, lab_sfm, conf_sfm = analyse(pts, est_poses)
    f_scan, planes_scan, _, _ = analyse(scan, ref_poses)

    dim_sfm, dim_scan = dimensions(planes_sfm, f_sfm), dimensions(planes_scan, f_scan)
    report = {
        "planes_sfm": [[p.name, p.support, round(p.sigma, 5)] for p in planes_sfm],
        "planes_scan": [[p.name, p.support, round(p.sigma, 5)] for p in planes_scan],
        "dimensions_sfm_units": dim_sfm,
        "dimensions_scan_m": dim_scan,
    }

    # Move SfM points into the scan frame (ground-truth poses: evaluation only).
    sim = umeyama(*camera_centres(est_poses, ref_poses))
    moved = sim.apply(pts)
    true_scale = sim.scale * 1.0  # SfM units -> scan metres

    # Which SfM axis corresponds to which scan axis (by direction, via the oracle rotation).
    dots = np.abs((sim.rotation @ f_sfm.axes).T @ f_scan.axes)
    axis_map = {i: int(dots[i].argmax()) for i in range(3)}

    def key(frame, axis):
        return "height" if axis == frame.up else f"horizontal_{axis}"

    # Each room dimension available in both clouds is a possible "tape measurement".
    report["tape_choices"] = []
    for i in range(3):
        k_sfm, k_scan = key(f_sfm, i), key(f_scan, axis_map[i])
        if k_sfm in dim_sfm and k_scan in dim_scan:
            scale_i = dim_scan[k_scan] / dim_sfm[k_sfm]
            report["tape_choices"].append({
                "dimension": k_sfm, "tape_m": dim_scan[k_scan], "scale_m_per_unit": scale_i,
                "scale_error_vs_true": scale_i / true_scale - 1.0,
            })
    report["true_scale_m_per_unit"] = true_scale
    usable = [c for c in report["tape_choices"] if c["dimension"] != "height"] or report["tape_choices"]
    scale = usable[0]["scale_m_per_unit"] if usable else float("nan")

    # Label agreement with scan-derived planes (evaluation-only oracle alignment).
    lab_ref, conf_ref = label_points(moved, f_scan, planes_scan, min_sigma=EVAL_TOLERANCE_M / 3)
    # Where do the SfM floor/ceiling planes lie in the scan frame, next to the scan's own?
    up_axis = f_scan.axes[:, f_scan.up]
    report["vertical_check_scan_frame_m"] = {
        "scan_floor": next((p.offset for p in planes_scan if p.role == "floor"), None),
        "scan_ceiling": next((p.offset for p in planes_scan if p.role == "ceiling"), None),
        "sfm_floor": float(np.median(moved[lab_sfm == "floor"] @ up_axis)) if (lab_sfm == "floor").any() else None,
        "sfm_ceiling": float(np.median(moved[lab_sfm == "ceiling"] @ up_axis)) if (lab_sfm == "ceiling").any() else None,
    }
    a, b = coarse(lab_sfm), coarse(lab_ref)
    agree = a == b
    report["label_agreement"] = {
        "overall": float(agree.mean()),
        "by_class_in_reference": {
            c: {"n": int((b == c).sum()), "agreement": float(agree[b == c].mean())}
            for c in sorted(set(b))
        },
    }
    bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]
    report["agreement_by_confidence"] = {
        f"{lo:.1f}-{min(hi, 1.0):.1f}": {
            "n": int(((conf_sfm >= lo) & (conf_sfm < hi)).sum()),
            "agreement": float(agree[(conf_sfm >= lo) & (conf_sfm < hi)].mean())
            if ((conf_sfm >= lo) & (conf_sfm < hi)).any() else None,
        }
        for lo, hi in bins
    }

    if np.isfinite(scale):
        graph = build_scene_graph(pts, lab_sfm, conf_sfm, f_sfm, planes_sfm, scale=scale, unit="m")
        report["scene_graph"] = graph
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    summary = {k: v for k, v in report.items() if k not in ("scene_graph",)}
    print(json.dumps(summary, indent=2, default=str))
    if "scene_graph" in report:
        g = report["scene_graph"]
        print("scene graph:", len(g["nodes"]), "nodes,", len(g["edges"]), "edges,",
              "labels", g["label_counts"])


if __name__ == "__main__":
    main()
