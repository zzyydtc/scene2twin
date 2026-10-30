"""Run SAM on a COLMAP model's images and lift the instances to its 3D points.

Usage (from the repo root):
    python -m semantics.run_sam --model outputs/office_fisheye/best \
        --images data/eth3d/office/images/dslr_images \
        --checkpoint data/models/sam_vit_b_01ec64.pth --out outputs/office_fisheye/sam
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from eval.colmap_io import read_model_poses
from semantics.evaluate import analyse
from semantics.lift import assign_mask_ids, cluster_instances
from semantics.scene_graph import cluster_objects


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two labelings of the same items."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2
    sum_cells = comb(table).sum()
    sum_rows, sum_cols = comb(table.sum(1)).sum(), comb(table.sum(0)).sum()
    expected = sum_rows * sum_cols / comb(len(a))
    maximum = (sum_rows + sum_cols) / 2
    return float((sum_cells - expected) / (maximum - expected)) if maximum != expected else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--long-side", type=int, default=1536)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rec = pycolmap.Reconstruction(str(args.model))
    pids = list(rec.points3D.keys())
    row_of = {pid: i for i, pid in enumerate(pids)}
    pts = np.array([rec.points3D[p].xyz for p in pids])

    obs: dict[int, list[tuple[int, np.ndarray]]] = {}
    for pid in pids:
        for el in rec.points3D[pid].track.elements:
            xy = rec.images[el.image_id].points2D[el.point2D_idx].xy
            obs.setdefault(el.image_id, []).append((row_of[pid], np.asarray(xy, dtype=float)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(device)
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    view_ids, overlays, per_image = [], {}, []
    for image_id, items in sorted(obs.items()):
        image = rec.images[image_id]
        bgr = cv2.imread(str(args.images / image.name))
        if bgr is None:
            raise FileNotFoundError(args.images / image.name)
        scale = args.long_side / max(bgr.shape[:2])
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        masks = generator.generate(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        rows = np.array([r for r, _ in items])
        xy = np.array([p for _, p in items]) * scale
        ids = assign_mask_ids(masks, small.shape[:2], xy)
        view_ids.append((rows, ids))
        per_image.append({"image": image.name, "masks": len(masks), "observations": len(items),
                          "with_mask": int((ids >= 0).sum()), "seconds": round(time.perf_counter() - t0, 1)})
        print(per_image[-1], flush=True)
        overlays[image.name] = (small, masks, rows, xy, ids)

    inst = cluster_instances(view_ids, len(pts))
    n_inst = int(inst.labels.max()) + 1

    # Compare with the structural stage (planes + DBSCAN object clusters).
    frame, planes, struct_labels, struct_conf = analyse(pts, read_model_poses(args.model))
    extent = float(np.ptp(pts @ frame.axes, axis=0).max())
    dbscan = np.full(len(pts), -1)
    for k, idx in enumerate(cluster_objects(pts, struct_labels, frame, extent)):
        dbscan[idx] = k
    both = (inst.labels >= 0) & (dbscan >= 0)

    instances = []
    for k in range(n_inst):
        sel = inst.labels == k
        c = pts[sel] @ frame.axes
        instances.append({
            "id": k, "points": int(sel.sum()), "max_views": int(inst.n_views[sel].max()),
            "mean_link_affinity": round(inst.edge_affinity[k], 3),
            "fraction_labelled_object": round(float((struct_labels[sel] == "object").mean()), 3),
            "size_sfm_units": [round(float(v), 3) for v in np.ptp(c, axis=0)],
        })
    report = {
        "images": len(per_image), "points": int(len(pts)),
        "points_in_instances": int((inst.labels >= 0).sum()), "instances": n_inst,
        "agreement_with_dbscan_ari": adjusted_rand_index(inst.labels[both], dbscan[both]) if both.sum() > 10 else None,
        "points_in_both": int(both.sum()),
        "instance_list": instances, "per_image": per_image,
    }
    (args.out / "sam_instances.json").write_text(json.dumps(report, indent=2))
    np.save(args.out / "instance_labels.npy", inst.labels)

    # Overlays for a visual check: mask outlines and points coloured by instance.
    rng = np.random.default_rng(0)
    palette = rng.integers(60, 255, size=(max(n_inst, 1), 3))
    for name in list(overlays)[:: max(1, len(overlays) // 3)][:3]:
        small, masks, rows, xy, ids = overlays[name]
        canvas = small.copy()
        for m in masks:
            cnts, _ = cv2.findContours(m["segmentation"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
        for (r, (x, y)) in zip(rows, xy):
            lab = inst.labels[r]
            colour = (128, 128, 128) if lab < 0 else tuple(int(v) for v in palette[lab])
            cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 0), -1)
            cv2.circle(canvas, (int(x), int(y)), 4, colour, -1)
        cv2.imwrite(str(args.out / f"overlay_{Path(name).stem}.png"), canvas)

    print(json.dumps({k: v for k, v in report.items() if k not in ("instance_list", "per_image")}, indent=2))
    for ins in instances[:12]:
        print(ins)


if __name__ == "__main__":
    main()
