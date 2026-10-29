"""Day 1: run a baseline SfM reconstruction on a folder of images.

Usage:
    python scripts/day1_baseline.py --images data/room/images --out outputs/room

Uses pycolmap for feature extraction, matching and incremental mapping.
This is the library baseline that later stages are compared against.
"""
import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    database = args.out / "database.db"
    sparse = args.out / "sparse"
    sparse.mkdir(exist_ok=True)

    pycolmap.extract_features(database, args.images)
    pycolmap.match_exhaustive(database)
    maps = pycolmap.incremental_mapping(database, args.images, sparse)

    if not maps:
        raise SystemExit("Reconstruction failed: no model produced.")
    for idx, rec in maps.items():
        print(f"model {idx}: {rec.num_reg_images()} images, "
              f"{rec.num_points3D()} points, "
              f"mean reprojection error {rec.compute_mean_reprojection_error():.3f} px")


if __name__ == "__main__":
    main()
