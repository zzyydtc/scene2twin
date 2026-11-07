"""Learned feature matching (DISK keypoints + LightGlue) written into a COLMAP database.

COLMAP's own pipeline detects SIFT keypoints and matches them with a ratio test. This module
replaces that step: DISK finds the keypoints, LightGlue matches every image pair, and the result is
stored in a COLMAP database so the same mappers and the same evaluation run on it unchanged.
COLMAP still does the geometric verification, so a pair is only kept when its matches fit a
two-view geometry.

Both networks are pretrained (DISK and LightGlue, through kornia); nothing is trained here.
"""
import itertools
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def list_images(images: Path) -> list[str]:
    return sorted(p.name for p in Path(images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def resize_scale(width: int, height: int, max_side: int) -> float:
    """Factor that brings the longer image side down to ``max_side`` (never enlarges)."""
    return min(1.0, max_side / max(width, height))


def to_colmap_keypoints(xy_resized: np.ndarray, scale: float) -> np.ndarray:
    """Network pixel coordinates (centre of the first pixel = 0) to COLMAP's (centre = 0.5) at full size."""
    return ((xy_resized + 0.5) / scale).astype(np.float32)


def extract_disk(images: Path, names: list[str], max_side: int, max_keypoints: int, device: str) -> dict:
    """DISK keypoints and descriptors per image: name -> (xy at full size, descriptors, image size)."""
    from kornia.feature import DISK

    model = DISK.from_pretrained("depth").to(device).eval()
    out = {}
    for name in names:
        bgr = cv2.imread(str(Path(images) / name))
        height, width = bgr.shape[:2]
        scale = resize_scale(width, height, max_side)
        small = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        # DISK needs the size to be a multiple of 16; crop the few missing pixels.
        h16, w16 = small.shape[0] // 16 * 16, small.shape[1] // 16 * 16
        small = small[:h16, :w16]
        tensor = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).to(device)
        with torch.no_grad():
            feats = model(tensor, n=max_keypoints, window_size=5, score_threshold=0.0)[0]
        out[name] = {
            "xy": to_colmap_keypoints(feats.keypoints.cpu().numpy(), scale),
            "xy_resized": feats.keypoints.cpu(),
            "descriptors": feats.descriptors.cpu(),
            "size": (w16, h16),
        }
    return out


def match_pairs(features: dict, names: list[str], device: str, min_matches: int = 15) -> dict:
    """LightGlue matches for every image pair: (i, j) -> uint32 array [m, 2] of keypoint indices."""
    from kornia.feature import LightGlue

    matcher = LightGlue("disk").to(device).eval()
    matches = {}
    for i, j in itertools.combinations(range(len(names)), 2):
        a, b = features[names[i]], features[names[j]]
        with torch.no_grad():
            result = matcher(
                {"image0": {"keypoints": a["xy_resized"][None].to(device), "descriptors": a["descriptors"][None].to(device),
                            "image_size": torch.tensor([a["size"]], dtype=torch.float32, device=device)},
                 "image1": {"keypoints": b["xy_resized"][None].to(device), "descriptors": b["descriptors"][None].to(device),
                            "image_size": torch.tensor([b["size"]], dtype=torch.float32, device=device)}}
            )
        idx = result["matches"][0].cpu().numpy().astype(np.uint32)
        if len(idx) >= min_matches:
            matches[(i, j)] = idx
    return matches


def build_database(images: Path, database: Path, camera_model: str, max_side: int = 1600,
                   max_keypoints: int = 4096, device: str | None = None) -> dict:
    """Create a COLMAP database with DISK keypoints and LightGlue matches, verified geometrically.

    Cameras and images are created by COLMAP's reader (one shared camera, as the controller uses),
    its SIFT keypoints are then replaced. Returns counts for the log.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    database = Path(database)
    if database.exists():
        database.unlink()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    sift = pycolmap.SiftExtractionOptions()
    sift.max_num_features = 16  # placeholder keypoints, replaced below; this call is only for cameras and image rows
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=pycolmap.FeatureExtractionOptions(sift=sift))

    names = list_images(images)
    features = extract_disk(images, names, max_side, max_keypoints, device)
    matches = match_pairs(features, names, device)

    db = pycolmap.Database.open(str(database))
    try:
        image_id = {im.name: im.image_id for im in db.read_all_images()}
        db.clear_keypoints()
        db.clear_descriptors()
        for name in names:
            db.write_keypoints(image_id[name], features[name]["xy"])
        for (i, j), idx in matches.items():
            db.write_matches(image_id[names[i]], image_id[names[j]], idx)
    finally:
        db.close()

    pairs_file = database.with_suffix(".pairs.txt")
    pairs_file.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in matches))
    pycolmap.verify_matches(database, pairs_file)

    db = pycolmap.Database.open(str(database))
    try:
        n_verified = db.num_verified_image_pairs
        n_verified = n_verified() if callable(n_verified) else n_verified
    finally:
        db.close()
    return {"n_images": len(names), "n_keypoints_mean": float(np.mean([len(f["xy"]) for f in features.values()])),
            "n_pairs_matched": len(matches), "n_pairs_verified": int(n_verified)}


def main() -> None:
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(description="Build a COLMAP database with DISK keypoints and LightGlue matches.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--max-side", type=int, default=1600, help="longer image side the network sees")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    args = parser.parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    info = build_database(args.images, args.database, args.camera_model, args.max_side, args.max_keypoints)
    info["seconds"] = round(time.perf_counter() - start, 1)
    info.update(max_side=args.max_side, max_keypoints=args.max_keypoints)
    print(json.dumps(info))
    args.database.with_suffix(".json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
