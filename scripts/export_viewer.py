"""Export a controller run to the static web viewer's data format.

Usage (from the repo root):
    python -m scripts.export_viewer --report outputs/office_agent/report.json \
        --name office --instance-labels outputs/office_agent/sam/instance_labels.npy \
        --images data/eth3d/office/images/dslr_images

Writes viewer/data/<name>/ with flat binary arrays (one value per point, same order),
downscaled photos in images/, the triangulated 2D keypoints of every photo (obs_*), and
a meta.json holding the planes, cameras, quality gates and the controller's decision log.
Coordinates are rotated into the room frame with +y up, the floor at y = 0, and centred
horizontally. They are in metres when the controller derived a scale, else in
reconstruction units; meta.json says which.

With --scene (an ETH3D scene folder) it also writes precision.f32 and error.f32, the predicted
and observed error of each point against the laser scan (NaN for points off the scan), and,
with --placement, scan.f32 (the scan around the reconstruction) plus one scan-to-viewer matrix
per registration result, so the viewer can show the scan placed by each method:

    --scene data/eth3d/office \
        --placement raster:outputs/placement/office.json:ours_raster_prior \
        --placement baseline:outputs/placement/office.json:ours_manhattan
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from eval.calibration import actual_error, evaluate, point_geometry, predicted_sigma, scan_normals
from eval.colmap_io import centre_from_pose, load_reference_scan, read_model_poses, read_reference_poses
from eval.metrics import pose_errors
from registration.sim3 import Sim3
from semantics.evaluate import analyse

PLACEMENT_LABELS = {
    "raster": "Occupancy-grid correlation + point-to-plane ICP (this work)",
    "baseline": "Room axes (Day 3 baseline)",
}


def room_to_view(coords: np.ndarray, up: int) -> np.ndarray:
    """Permute room-frame coordinates so the vertical axis becomes y (cyclic, keeps handedness)."""
    order = [(up + 2) % 3, up, (up + 1) % 3]
    return coords[..., order]


def affine_matrix(f) -> np.ndarray:
    """Row-major 4x4 matrix of an affine map ``f`` on (N, 3) points."""
    b = f(np.zeros((1, 3)))[0]
    m = np.eye(4)
    m[:3, :3] = (f(np.eye(3)) - b).T
    m[:3, 3] = b
    return m


def scan_to_view_matrix(placement: Sim3, to_view) -> np.ndarray:
    """Matrix taking scan-frame points to viewer coordinates for a registration result.

    ``placement`` maps reconstruction coordinates into the scan frame, so a scan point is first
    brought back into the reconstruction frame and then through the same mapping as the points.
    """
    def f(y: np.ndarray) -> np.ndarray:
        return to_view(((y - placement.translation) / placement.scale) @ placement.rotation)
    return affine_matrix(f)


def read_placement(spec: str) -> tuple[str, Sim3, dict]:
    """``name:file:key`` -> (name, transform, the run's recorded metrics)."""
    name, path, key = spec.split(":", 2)
    run = json.loads(Path(path).read_text())[key][0]
    t = run["transform"]
    return name, Sim3(float(t["scale"]), np.array(t["rotation"]), np.array(t["translation"])), run


def crop_scan(scan: np.ndarray, box: tuple[np.ndarray, np.ndarray], margin: float, limit: int = 100_000) -> np.ndarray:
    """Scan points within ``margin`` of a box, voxel-thinned until at most ``limit`` remain."""
    import open3d as o3d
    lo, hi = box[0] - margin, box[1] + margin
    near = scan[np.all((scan >= lo) & (scan <= hi), axis=1)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(near))
    for voxel in (0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.9):
        thin = np.asarray(pc.voxel_down_sample(voxel).points)
        if len(thin) <= limit:
            return thin
    return thin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True, help="report.json from agent.controller")
    parser.add_argument("--name", required=True)
    parser.add_argument("--instance-labels", type=Path, help="instance_labels.npy from semantics.run_sam")
    parser.add_argument("--out", type=Path, default=Path("viewer/data"))
    parser.add_argument("--images", type=Path, help="folder with the original photos; downscaled copies are exported")
    parser.add_argument("--thumb-width", type=int, default=800)
    parser.add_argument("--scene", type=Path, help="ETH3D scene folder; adds per-point precision and error")
    parser.add_argument("--placement", action="append", default=[], metavar="NAME:FILE:KEY",
                        help="registration result from registration.compare (needs --scene); NAME is raster or baseline")
    args = parser.parse_args()
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)

    report =json.loads(args.report.read_text())
    model_dir = args.report.parent / report["chosen_model"].lower() / "best"
    rec = pycolmap.Reconstruction(str(model_dir))
    pts3d = list(rec.points3D.values())  # same order as eval.colmap_io.read_model_points
    xyz = np.array([p.xyz for p in pts3d], dtype=np.float64)
    rgb = np.array([p.color for p in pts3d], dtype=np.uint8)
    poses = read_model_poses(model_dir)

    frame, planes, labels, conf = analyse(xyz, poses)
    scale = report["metric_scale_m_per_unit"]
    unit = "m" if scale else "units"
    k = float(scale or 1.0)

    # Room frame -> viewer frame (+y up), scaled, floor at y = 0, centred horizontally.
    view = room_to_view(xyz @ frame.axes, frame.up) * k
    floor = next((p for p in planes if p.role == "floor"), None)
    y0 = floor.offset * k if floor else float(np.percentile(view[:, 1], 0.5))
    shift = np.array([np.median(view[:, 0]), y0, np.median(view[:, 2])])
    view = view - shift

    names = ["object"] + sorted({p.name for p in planes})
    label_idx = np.array([names.index(str(l)) for l in labels], dtype=np.uint8)

    instance = np.full(len(xyz), -1, dtype=np.int16)
    instances = []
    if args.instance_labels:
        inst = np.load(args.instance_labels)
        if len(inst) == len(xyz):
            instance = inst.astype(np.int16)
            excluded = set(report["excluded_instances"])
            for i in np.unique(inst[inst >= 0]):
                instances.append({"id": int(i), "points": int((inst == i).sum()), "excluded": int(i) in excluded})
        else:
            print(f"instance labels cover {len(inst)} points, model has {len(xyz)}: skipped")

    # Photos: a downscaled copy of each registered image, the pose and intrinsics, and the
    # 2D keypoints that were triangulated into 3D points (so the viewer can overlay the
    # model on the photo exactly, without having to model lens distortion).
    point_index = {pid: i for i, pid in enumerate(rec.points3D.keys())}
    by_name = {Path(im.name).name: im for im in rec.images.values()}
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    obs_idx, obs_u, obs_v, cams = [], [], [], []
    for name, (r, t) in sorted(poses.items()):
        im = by_name[name]
        cam = rec.cameras[im.camera_id]
        c = centre_from_pose(r, t)
        cv = room_to_view(c @ frame.axes, frame.up) * k - shift
        fwd = room_to_view((r.T @ np.array([0.0, 0.0, 1.0])) @ frame.axes, frame.up)
        up_vec = room_to_view((r.T @ np.array([0.0, -1.0, 0.0])) @ frame.axes, frame.up)  # COLMAP y points down
        first = len(obs_idx)
        for p2 in im.points2D:
            if p2.has_point3D() and p2.point3D_id in point_index:
                obs_idx.append(point_index[p2.point3D_id])
                obs_u.append(p2.xy[0] / cam.width)
                obs_v.append(p2.xy[1] / cam.height)
        thumb = f"{Path(name).stem}.jpg"
        if args.images:
            src = cv2.imread(str(args.images / name))
            if src is not None:
                scale_px = args.thumb_width / src.shape[1]
                small = cv2.resize(src, None, fx=scale_px, fy=scale_px, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(img_out / thumb), small, [cv2.IMWRITE_JPEG_QUALITY, 72])
        cams.append({
            "name": name, "image": thumb if args.images else None,
            "position": cv.round(4).tolist(), "forward": fwd.round(4).tolist(), "up": up_vec.round(4).tolist(),
            "width": int(cam.width), "height": int(cam.height), "focal_y": float(cam.focal_length_y),
            "obs_offset": first, "obs_count": len(obs_idx) - first,
        })

    plane_meta = []
    for p in planes:
        slot = [(frame.up + 2) % 3, frame.up, (frame.up + 1) % 3].index(p.axis_index)
        mean_conf = float(conf[labels == p.name].mean()) if (labels == p.name).any() else 0.0
        offset = (p.offset * k - shift[slot])
        plane_meta.append({
            "name": p.name, "role": p.role, "axis": slot, "offset": round(float(offset), 4),
            "support_points": p.support, "sigma": round(float(p.sigma * k), 4),
            "mean_confidence": round(mean_conf, 3),
        })

    def to_view(coords: np.ndarray) -> np.ndarray:
        return room_to_view(coords @ frame.axes, frame.up) * k - shift

    extras = {}
    if args.scene:
        pose = pose_errors(read_model_poses(model_dir), read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt"))
        geo = point_geometry(model_dir)
        scan = load_reference_scan(args.scene / "scan_clean", voxel=0.02)
        normals = scan_normals(scan)
        cal = evaluate(geo, pose.transform, scan, normals)
        err, n, kept = actual_error(pose.transform.apply(geo.xyz), scan, normals)
        sigma = predicted_sigma(geo, pose.transform, n, cal["sigma_px"])
        err[~kept] = np.nan
        sigma[~kept] = np.nan
        err.astype("<f4").tofile(out / "error.f32")
        sigma.astype("<f4").tofile(out / "precision.f32")
        extras["calibration"] = {
            "sigma_px": round(cal["sigma_px"], 3), "n_on_scan": cal["n_on_scan"],
            "predicted_median_m": cal["predicted_sigma_m"]["median"], "observed_median_m": cal["observed_error_m"]["median"],
            "observed_p95_m": cal["observed_error_m"]["p95"], "inflation": round(cal["inflation_factor"], 2),
            "coverage_raw": {key: round(v, 3) for key, v in cal["coverage_raw"].items()},
            "cap_m": float(np.nanpercentile(err, 95)),
        }
        if args.placement:
            placements = [read_placement(spec) for spec in args.placement]
            ref = next((t for name, t, _ in placements if name == "raster"), placements[0][1]).apply(xyz)
            thin = crop_scan(scan, (ref.min(axis=0), ref.max(axis=0)), margin=2.0)
            thin.astype("<f4").tofile(out / "scan.f32")
            extras["scan"] = {
                "n_points": int(len(thin)),
                "placements": [{
                    "id": name, "label": PLACEMENT_LABELS.get(name, name),
                    "camera_rmse_m": run["camera_rmse_m"], "median_m": run["median_m"], "within_2cm": run["within_2cm"],
                    "scale": run["scale"], "seconds": run.get("seconds"),
                    "matrix": scan_to_view_matrix(t, to_view).round(6).ravel().tolist(),
                } for name, t, run in placements],
            }

    lo, hi = np.percentile(view, [0.5, 99.5], axis=0)
    meta = {
        "name": args.name, "unit": unit, "scale": scale, "scale_basis": report["metric_scale_basis"],
        "n_points": int(len(xyz)), "labels": names, "planes": plane_meta, "cameras": cams,
        "bounds": {"min": lo.round(3).tolist(), "max": hi.round(3).tolist()},
        "chosen_model": report["chosen_model"], "candidates": report["candidates"],
        "gates": report["gates"], "decision_log": report["decision_log"],
        "dimensions_units": report["dimensions_units"], "instances": instances, **extras,
    }

    view.astype("<f4").tofile(out / "points.f32")
    np.array(obs_idx, dtype="<u4").tofile(out / "obs_index.u32")
    np.array(obs_u, dtype="<f4").tofile(out / "obs_u.f32")
    np.array(obs_v, dtype="<f4").tofile(out / "obs_v.f32")
    rgb.tofile(out / "colors.u8")
    label_idx.tofile(out / "labels.u8")
    np.round(np.clip(conf, 0, 1) * 255).astype(np.uint8).tofile(out / "confidence.u8")
    instance.astype("<i2").tofile(out / "instances.i16")
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{args.name}: {len(xyz)} points, {len(planes)} planes, {len(cams)} cameras, unit {unit}, {size/1024:.0f} KiB")


if __name__ == "__main__":
    main()
