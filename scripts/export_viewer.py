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
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from eval.colmap_io import centre_from_pose, read_model_poses
from semantics.evaluate import analyse


def room_to_view(coords: np.ndarray, up: int) -> np.ndarray:
    """Permute room-frame coordinates so the vertical axis becomes y (cyclic, keeps handedness)."""
    order = [(up + 2) % 3, up, (up + 1) % 3]
    return coords[..., order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True, help="report.json from agent.controller")
    parser.add_argument("--name", required=True)
    parser.add_argument("--instance-labels", type=Path, help="instance_labels.npy from semantics.run_sam")
    parser.add_argument("--out", type=Path, default=Path("viewer/data"))
    parser.add_argument("--images", type=Path, help="folder with the original photos; downscaled copies are exported")
    parser.add_argument("--thumb-width", type=int, default=800)
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

    lo, hi = np.percentile(view, [0.5, 99.5], axis=0)
    meta = {
        "name": args.name, "unit": unit, "scale": scale, "scale_basis": report["metric_scale_basis"],
        "n_points": int(len(xyz)), "labels": names, "planes": plane_meta, "cameras": cams,
        "bounds": {"min": lo.round(3).tolist(), "max": hi.round(3).tolist()},
        "chosen_model": report["chosen_model"], "candidates": report["candidates"],
        "gates": report["gates"], "decision_log": report["decision_log"],
        "dimensions_units": report["dimensions_units"], "instances": instances,
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
