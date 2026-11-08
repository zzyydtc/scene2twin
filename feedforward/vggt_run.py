"""Feed-forward reconstruction with VGGT, scored with the same metrics as the SfM pipeline.

VGGT (Wang et al., CVPR 2025) is a pretrained network that takes a set of images and, in one
forward pass, returns a camera (pose and field of view), a depth map and a confidence map for each
image. There is no feature matching, no triangulation and no bundle adjustment. This module runs it
and scores its cameras and its points against the ETH3D reference poses and laser scan exactly as
``eval.run_eval`` scores a COLMAP model.

The VGGT code is not part of this repository; it is cloned to ``data/third_party/vggt`` and the
weights are downloaded to ``data/models/vggt-1b`` (see the README). Nothing is trained here.

Usage (from the repo root):
    python -m feedforward.vggt_run --images data/eth3d/office/images/dslr_images \
        --scene data/eth3d/office --out outputs/vggt/office
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
VGGT_CODE = Path("data/third_party/vggt")
VGGT_WEIGHTS = Path("data/models/vggt-1b/model.safetensors")


def extrinsics_to_poses(names: list[str], extrinsic: np.ndarray) -> dict:
    """VGGT camera-from-world matrices [S, 3, 4] to the (R, t) dict the metrics take, keyed by file name."""
    return {name: (np.asarray(m[:, :3], dtype=np.float64), np.asarray(m[:, 3], dtype=np.float64)) for name, m in zip(names, extrinsic)}


def focal_in_original_pixels(intrinsic: np.ndarray, network_size: tuple[int, int], original_size: tuple[int, int]) -> tuple[float, float]:
    """Median focal lengths (fx, fy) of VGGT's intrinsics, converted from the network image to the photo.

    ``network_size`` and ``original_size`` are (width, height). The network image is the photo resized
    to a different width and height factor, so x and y are scaled separately.
    """
    fx = float(np.median(intrinsic[:, 0, 0])) * original_size[0] / network_size[0]
    fy = float(np.median(intrinsic[:, 1, 1])) * original_size[1] / network_size[1]
    return fx, fy


def confident_points(points: np.ndarray, confidence: np.ndarray, keep_fraction: float, max_points: int, seed: int = 0) -> np.ndarray:
    """The ``keep_fraction`` most confident points of a [S, H, W, 3] cloud, thinned to at most ``max_points``.

    Confidence is ranked over the whole set of images, not per image, so a confident image keeps more.
    """
    xyz = points.reshape(-1, 3)
    conf = confidence.reshape(-1)
    ok = np.isfinite(xyz).all(axis=1) & np.isfinite(conf)
    xyz, conf = xyz[ok], conf[ok]
    if len(xyz) == 0:
        return xyz
    keep = conf >= np.quantile(conf, 1.0 - keep_fraction)
    xyz = xyz[keep]
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)]
    return xyz


def load_model(device: str):
    """VGGT-1B from the local clone and the local weights file."""
    import torch
    from safetensors.torch import load_file

    sys.path.insert(0, str(VGGT_CODE.resolve()))
    from vggt.models.vggt import VGGT

    model = VGGT()
    model.load_state_dict(load_file(str(VGGT_WEIGHTS)))
    return model.to(device).eval()


def predict(images_dir: Path, names: list[str], device: str) -> dict:
    """Run VGGT on all images at once; return numpy arrays plus timing and peak GPU memory."""
    import torch

    sys.path.insert(0, str(VGGT_CODE.resolve()))
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = load_model(device)
    images = load_and_preprocess_images([str(Path(images_dir) / n) for n in names]).to(device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
        tokens, patch_start = model.aggregator(images[None])
        pose_enc = model.camera_head(tokens)[-1]
        depth, depth_conf = model.depth_head(tokens, images[None], patch_start)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.float(), images.shape[-2:])
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    extrinsic, intrinsic = extrinsic[0].cpu().numpy(), intrinsic[0].cpu().numpy()
    depth, depth_conf = depth[0].float().cpu().numpy(), depth_conf[0].float().cpu().numpy()
    return {
        "extrinsic": extrinsic, "intrinsic": intrinsic, "depth": depth, "depth_conf": depth_conf,
        "world_points": unproject_depth_map_to_point_map(depth, extrinsic, intrinsic),
        "network_size": (int(images.shape[-1]), int(images.shape[-2])), "seconds": seconds,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
    }


def main() -> None:
    import torch

    from eval.colmap_io import load_reference_scan, read_reference_poses
    from eval.metrics import accuracy_completeness, pose_errors

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True, help="ETH3D scene folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", default="0.5,0.1", help="fractions of the most confident points to score")
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--voxel", type=float, default=0.02, help="reference scan voxel size (m)")
    args = parser.parse_args()

    names = sorted(p.name for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = predict(args.images, names, device)

    ref_poses = read_reference_poses(args.scene / "dslr_calibration_jpg" / "images.txt")
    scan = load_reference_scan(args.scene / "scan_clean", voxel=args.voxel)
    pose = pose_errors(extrinsics_to_poses(names, pred["extrinsic"]), ref_poses)

    import cv2
    height, width = cv2.imread(str(args.images / names[0])).shape[:2]
    fx, fy = focal_in_original_pixels(pred["intrinsic"], pred["network_size"], (width, height))
    ref_cam = (args.scene / "dslr_calibration_jpg" / "cameras.txt").read_text().strip().splitlines()[-1].split()
    result = {"n_images": len(names), "network_size": pred["network_size"], "seconds": round(pred["seconds"], 2),
              "peak_gpu_gb": pred["peak_gpu_gb"], "poses": pose.to_dict(),
              "focal_px": {"vggt_fx": fx, "vggt_fy": fy, "reference_fx": float(ref_cam[4]), "reference_fy": float(ref_cam[5])}, "points": {}}
    for keep in [float(k) for k in args.keep.split(",")]:
        xyz = confident_points(pred["world_points"], pred["depth_conf"], keep, args.max_points)
        acc = accuracy_completeness(pose.transform.apply(xyz), scan)
        result["points"][str(keep)] = {"n": len(xyz), "accuracy_median_m": acc["accuracy_median_m"], "precision_2cm": acc["at_threshold"]["0.02"]["precision"]}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "predictions.npz", extrinsic=pred["extrinsic"], intrinsic=pred["intrinsic"], depth_conf=pred["depth_conf"].astype(np.float16))
    (args.out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
