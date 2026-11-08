"""MASt3R-SfM: pairwise network predictions turned into one reconstruction, scored like the others.

MASt3R (Leroy et al., 2024) is a pretrained network that, for a pair of images, predicts a 3D point for
every pixel in a shared frame plus a descriptor per pixel. MASt3R-SfM runs it on image pairs, matches
the descriptors, and aligns all pairs into one scene with an optimisation (``sparse_global_alignment``),
ending with a refinement of the 2D reprojection error. Unlike VGGT it is not a single pass over all
images, and it optimises cameras, depths and (here) one shared focal length.

The MASt3R code is not part of this repository; it is cloned (with its dust3r submodule) to
``data/third_party/mast3r`` and the weights are downloaded to ``data/models/mast3r-vitl-512``.

Usage (from the repo root):
    python -m feedforward.mast3r_run --images data/eth3d/office/images/dslr_images \
        --scene data/eth3d/office --out outputs/mast3r/office
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from feedforward.vggt_run import IMAGE_SUFFIXES, confident_points, extrinsics_to_poses

MAST3R_CODE = Path("data/third_party/mast3r")
MAST3R_WEIGHTS = Path("data/models/mast3r-vitl-512")


def cam_to_world_to_extrinsic(cam2w: np.ndarray) -> np.ndarray:
    """Camera-to-world matrices [S, 4, 4] to camera-from-world [S, 3, 4]."""
    out = np.empty((len(cam2w), 3, 4))
    for i, m in enumerate(cam2w):
        out[i] = np.linalg.inv(m)[:3]
    return out


def focal_in_photo_pixels(focal: float, network_width: int, original_width: int) -> float:
    """Focal length of the network image converted to the photo.

    MASt3R shrinks the photo keeping its aspect ratio (then crops a few rows or columns), so one factor,
    the width ratio, applies to both axes.
    """
    return float(focal) * original_width / network_width


def network_size_for(original_size: tuple[int, int], size: int = 512, patch: int = 16) -> tuple[int, int]:
    """(width, height) MASt3R's loader gives a photo: long side to ``size``, then a centred crop to a multiple of ``patch``."""
    width, height = original_size
    scale = size / max(width, height)
    w, h = round(width * scale), round(height * scale)
    return ((2 * (w // 2)) // patch * patch, (2 * (h // 2)) // patch * patch)


def inside_box(points: np.ndarray, reference: np.ndarray, margin: float) -> np.ndarray:
    """Finite points that lie within ``margin`` metres of the reference cloud's bounding box.

    Network outputs can contain a few absurdly distant points; they say nothing about accuracy and make
    nearest-neighbour scoring extremely slow.
    """
    lo, hi = reference.min(axis=0) - margin, reference.max(axis=0) + margin
    keep = np.isfinite(points).all(axis=1) & (points >= lo).all(axis=1) & (points <= hi).all(axis=1)
    return points[keep]


def run(images_dir: Path, names: list[str], out: Path, graph: str, device: str, size: int = 512,
        niter1: int = 500, niter2: int = 200) -> dict:
    """Run MASt3R-SfM on all images; return numpy results and timing."""
    import torch

    # MASt3R calls ``scipy.cluster.hierarchy.distance``, which newer scipy no longer exposes there.
    import scipy.cluster.hierarchy as hierarchy
    import scipy.spatial.distance as spatial_distance

    if not hasattr(hierarchy, "distance"):
        hierarchy.distance = spatial_distance
    root = MAST3R_CODE.resolve()
    sys.path[:0] = [str(root), str(root / "dust3r")]
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.model import AsymmetricMASt3R

    model = AsymmetricMASt3R.from_pretrained(str(MAST3R_WEIGHTS)).to(device).eval()
    files = [str(Path(images_dir) / n) for n in names]
    imgs = load_images(files, size=size, verbose=False)
    pairs = make_pairs(imgs, scene_graph=graph, prefilter=None, symmetrize=True)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    scene = sparse_global_alignment(files, pairs, str(cache), model, lr1=0.07, niter1=niter1, lr2=0.014, niter2=niter2,
                                    device=device, opt_depth=True, shared_intrinsics=True, matching_conf_thr=5.0)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    height, width = (int(v) for v in imgs[0]["true_shape"][0])
    pts, _, conf = scene.get_dense_pts3d(clean_depth=True)
    return {
        "cam2w": scene.get_im_poses().detach().cpu().numpy(), "focals": scene.get_focals().detach().cpu().numpy(),
        "network_size": (width, height),
        "sparse_points": [p.detach().cpu().numpy() for p in scene.get_sparse_pts3d()],
        "dense_points": np.stack([p.detach().cpu().numpy() for p in pts]), "dense_conf": np.stack([c.detach().cpu().numpy().ravel() for c in conf]),
        "n_pairs": len(pairs), "seconds": seconds,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
    }


def main() -> None:
    import cv2
    import torch

    from eval.colmap_io import load_reference_scan, read_reference_poses
    from eval.metrics import accuracy_completeness, pose_errors

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True, help="ETH3D scene folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graph", default="complete", help="dust3r scene graph: complete, swin-5, ...")
    parser.add_argument("--keep", default="0.5,0.1", help="fractions of the most confident dense points to score")
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--from-predictions", action="store_true", help="score the cameras saved in <out>/predictions.npz; skip the network and the points")
    args = parser.parse_args()

    names = sorted(p.name for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height, width = cv2.imread(str(args.images / names[0])).shape[:2]
    if args.from_predictions:
        saved = np.load(args.out / "predictions.npz")
        pred = {"cam2w": saved["cam2w"], "focals": saved["focals"], "network_size": network_size_for((width, height)),
                "n_pairs": None, "seconds": None, "peak_gpu_gb": None, "sparse_points": [], "dense_points": None, "dense_conf": None}
    else:
        pred = run(args.images, names, args.out, args.graph, device)

    # Save the expensive part first: scoring reads a large scan and can take long on a big scene.
    np.savez_compressed(args.out / "predictions.npz", extrinsic=cam_to_world_to_extrinsic(pred["cam2w"]), cam2w=pred["cam2w"], focals=pred["focals"])

    ref_poses = read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt")
    scan = load_reference_scan(args.scene / "scan_clean", voxel=0.02)
    extrinsic = cam_to_world_to_extrinsic(pred["cam2w"])
    pose = pose_errors(extrinsics_to_poses(names, extrinsic), ref_poses)

    fx = focal_in_photo_pixels(float(np.median(pred["focals"])), pred["network_size"][0], width)
    ref_cam = (args.scene / "dslr_calibration_jpg" / "cameras.txt").read_text().strip().splitlines()[-1].split()
    result = {"n_images": len(names), "graph": args.graph, "n_pairs": pred["n_pairs"], "network_size": pred["network_size"],
              "seconds": None if pred["seconds"] is None else round(pred["seconds"], 1), "peak_gpu_gb": pred["peak_gpu_gb"], "poses": pose.to_dict(),
              "focal_px": {"fx": fx, "fy": fx, "reference_fx": float(ref_cam[4]), "reference_fy": float(ref_cam[5])},
              "points": {}}
    if not args.from_predictions:
        def score_points(xyz: np.ndarray) -> dict:
            xyz = inside_box(pose.transform.apply(xyz), scan, margin=5.0)
            acc = accuracy_completeness(xyz, scan)
            return {"n": len(xyz), "accuracy_median_m": acc["accuracy_median_m"], "precision_2cm": acc["at_threshold"]["0.02"]["precision"]}

        sparse = np.concatenate(pred["sparse_points"])
        n_sparse = len(sparse)
        if n_sparse > args.max_points:
            sparse = sparse[np.random.default_rng(0).choice(n_sparse, args.max_points, replace=False)]
        result["sparse_points"] = score_points(sparse) | {"n_total": n_sparse}
        for keep in [float(k) for k in args.keep.split(",")]:
            result["points"][str(keep)] = score_points(confident_points(pred["dense_points"], pred["dense_conf"], keep, args.max_points))
    (args.out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
