"""Tools the controller calls: thin wrappers over the pipeline stages."""
from pathlib import Path

import pycolmap


def reconstruct(images: Path, out: Path, camera_model: str, reuse: bool = True) -> dict:
    """Run COLMAP SfM with one camera model and keep the largest model in ``out/best``.

    Returns image and point counts and the mean reprojection error. With ``reuse`` an
    existing ``out/best`` is loaded instead of recomputed.
    """
    best = out / "best"
    n_images = len([p for p in Path(images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if reuse and (best / "images.bin").exists():
        rec = pycolmap.Reconstruction(str(best))
        n_models = None  # not recorded for a reused model
    else:
        out.mkdir(parents=True, exist_ok=True)
        database = out / "database.db"
        if database.exists():
            database.unlink()
        sparse = out / "sparse"
        sparse.mkdir(exist_ok=True)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = camera_model
        pycolmap.extract_features(
            database, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader
        )
        pycolmap.match_exhaustive(database)
        maps = pycolmap.incremental_mapping(database, images, sparse)
        if not maps:
            return {"camera_model": camera_model, "path": None, "n_images": n_images,
                    "n_registered": 0, "n_models": 0, "n_points": 0, "reproj_px": float("inf")}
        best_idx = max(maps, key=lambda i: maps[i].num_reg_images())
        best.mkdir(exist_ok=True)
        maps[best_idx].write(best)
        rec = maps[best_idx]
        n_models = len(maps)
    return {
        "camera_model": camera_model, "path": str(best), "n_images": n_images,
        "n_registered": rec.num_reg_images(), "n_models": n_models,
        "n_points": rec.num_points3D(),
        "reproj_px": float(rec.compute_mean_reprojection_error()),
    }
