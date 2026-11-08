"""Export the camera poses of every reconstruction method on a scene, for the viewer's method comparison.

Each method's cameras are fitted to the ETH3D reference cameras with a Sim(3) (the same fit as the
camera error in ``eval.metrics``), then carried into the viewer's frame with the laser-scan placement
matrix that ``scripts/export_viewer.py`` already wrote to ``meta.json``. The reference cameras, which live
in the scan frame, go through the same matrix. Only poses are exported (a few kilobytes per scene).

The viewer's own cameras (the controller's model) are registered to the scan by that matrix, so the
reference cameras must land on them to within the registration error. This script checks that and stops
if they do not, because every drawn error would be wrong otherwise.

Usage (from the repo root, after ``scripts.export_viewer`` and the method runs of Days 8 to 11):
    python -m scripts.export_methods --scene office
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.colmap_io import centre_from_pose, read_model_poses, read_reference_poses
from eval.metrics import pose_errors, rotation_angle_deg
from feedforward.vggt_run import extrinsics_to_poses
from registration.sim3 import umeyama

# Second view of each method: the Sim(3) is fitted on the best fraction of the cameras only. A few badly
# placed cameras otherwise drag the fit and make every camera look wrong. The subset is chosen with the
# reference poses, so a user without a reference could not do this; the viewer says so.
TRIM_FRACTION = 0.8

# (id, label, where the poses are, note). A directory is a COLMAP model; a .npz holds a network's
# camera-from-world matrices in the order of the sorted image names.
METHODS = {
    "office": [
        ("sift_incremental", "SIFT, incremental mapper", "outputs/sfm_engines/office/incremental_seed1/best", "Day 8. The classical pipeline the viewer's points come from (principal point fixed)."),
        ("sift_global", "SIFT, global mapper", "outputs/sfm_engines/office/global_seed1/best", "Day 8, seed 1 of 5."),
        ("disk_lightglue", "DISK + LightGlue, incremental", "outputs/learned/office_s3200_k8192_sfm/incremental_seed1/best", "Day 9. Learned matcher, same mapper; 5 of the 26 photos were not registered."),
        ("vggt", "VGGT alone", "outputs/vggt/office/predictions.npz", "Day 10. One feed-forward pass, no matching."),
        ("vggt_colmap", "VGGT cameras, then COLMAP", "outputs/vggt/office_refined/best", "Day 10. VGGT poses as the start for triangulation and bundle adjustment."),
        ("vggt_colmap_pp", "VGGT, then COLMAP with principal point refined", "outputs/vggt/office_refined_pp/best", "Day 10. Mean reprojection error 0.38 px, lower than the correct model's, on a wrong model."),
        ("mast3r", "MASt3R-SfM alone", "outputs/mast3r/office/predictions.npz", "Day 11. Complete pair graph, 512 px."),
        ("mast3r_colmap", "MASt3R cameras, then COLMAP", "outputs/mast3r/office_refined/best", "Day 11."),
    ],
    "delivery_area": [
        ("sift_incremental", "SIFT, incremental mapper", "outputs/sfm_engines/delivery_area/incremental_seed1/best", "Day 8. The classical pipeline the viewer's points come from (principal point fixed)."),
        ("sift_global", "SIFT, global mapper", "outputs/sfm_engines/delivery_area/global_seed1/best", "Day 8, seed 1 of 5. This seed's model is wrong as a whole."),
        ("disk_lightglue", "DISK + LightGlue, incremental", "outputs/learned/delivery_s3200_all/incremental_seed1/best", "Day 9. Learned matcher, same mapper; DSC_0711 to 0714 were not registered."),
        ("vggt", "VGGT alone", "outputs/vggt/delivery_area/predictions.npz", "Day 10. One feed-forward pass, no matching."),
        ("vggt_colmap", "VGGT cameras, then COLMAP", "outputs/vggt/delivery_area_refined/best", "Day 10. VGGT poses as the start for triangulation and bundle adjustment."),
        ("vggt_colmap_pp", "VGGT, then COLMAP with principal point refined", "outputs/vggt/delivery_area_refined_pp/best", "Day 10."),
        ("mast3r", "MASt3R-SfM alone", "outputs/mast3r/delivery_area/predictions.npz", "Day 11. DSC_0711 to 0714 sit about 97 degrees off the other 40."),
        ("mast3r_colmap", "MASt3R cameras, then COLMAP", "outputs/mast3r/delivery_area_refined/best", "Day 11. The refinement repairs the other 40 cameras (1.35 cm) but leaves DSC_0711 to 0714 7 to 10 m off, at a reprojection error of 0.54 px."),
    ],
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def load_poses(source: Path, images_dir: Path) -> dict:
    if source.suffix == ".npz":
        names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        return extrinsics_to_poses(names, np.load(source)["extrinsic"])
    return read_model_poses(source)


def camera_vectors(pose: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, viewing direction and up direction of a camera-from-world pose (COLMAP: z forward, y down)."""
    r, t = pose
    return -r.T @ t, r.T @ np.array([0.0, 0.0, 1.0]), r.T @ np.array([0.0, -1.0, 0.0])


def to_view(matrix: np.ndarray, centre: np.ndarray, forward: np.ndarray, up: np.ndarray) -> dict:
    """Scan-frame camera into the viewer frame; directions keep unit length."""
    lin = matrix[:3, :3]
    unit = lambda v: (lin @ v) / np.linalg.norm(lin @ v)
    return {"position": (lin @ centre + matrix[:3, 3]).round(4).tolist(), "forward": unit(forward).round(4).tolist(), "up": unit(up).round(4).tolist()}


def trimmed_fit(poses: dict, ref: dict, fraction: float = TRIM_FRACTION, max_iter: int = 50):
    """Sim(3) fitted on the ``fraction`` of cameras that fit best (iterated until the subset stops changing)."""
    names = sorted(set(poses) & set(ref))
    c_est = np.array([centre_from_pose(*poses[n]) for n in names])
    c_ref = np.array([centre_from_pose(*ref[n]) for n in names])
    n_keep = max(3, int(np.ceil(fraction * len(names))))
    keep = np.arange(len(names))
    for _ in range(max_iter):
        sim = umeyama(c_est[keep], c_ref[keep], with_scale=True)
        residual = np.linalg.norm(sim.apply(c_est) - c_ref, axis=1)
        new = np.sort(np.argsort(residual)[:n_keep])
        if np.array_equal(new, keep):
            break
        keep = new
    return sim, {names[i] for i in keep}


def place_cameras(poses: dict, ref: dict, matrix: np.ndarray, sim, fitted: set | None = None) -> list[dict]:
    """Cameras of one method in the viewer frame, each with its position and rotation error against the reference."""
    cameras = []
    for name, pose in sorted(poses.items()):
        if name not in ref:
            continue
        centre, forward, up = camera_vectors(pose)
        aligned = sim.apply(centre[None])[0]
        entry = to_view(matrix, aligned, sim.rotation @ forward, sim.rotation @ up)
        r_aligned = pose[0] @ sim.rotation.T
        entry.update(name=name, error_m=round(float(np.linalg.norm(aligned - camera_vectors(ref[name])[0])), 5),
                     rotation_error_deg=round(float(rotation_angle_deg(r_aligned @ ref[name][0].T)), 3))
        if fitted is not None:
            entry["fitted"] = name in fitted
        cameras.append(entry)
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(METHODS), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/eth3d"))
    parser.add_argument("--viewer-data", type=Path, default=Path("viewer/data"))
    parser.add_argument("--tolerance", type=float, default=0.002, help="largest accepted difference (m) between the RMS gap measured here and the registration error recorded in meta.json")
    args = parser.parse_args()

    scene_dir = args.data / args.scene
    images_dir = scene_dir / "images" / "dslr_images"
    out_dir = args.viewer_data / args.scene
    meta = json.loads((out_dir / "meta.json").read_text())
    placement = next(p for p in meta["scan"]["placements"] if p["id"] == "raster")
    matrix = np.array(placement["matrix"]).reshape(4, 4)
    ref = read_reference_poses(scene_dir / "dslr_calibration_jpg" / "images.txt")

    reference = {}
    for name, pose in ref.items():
        reference[name] = to_view(matrix, *camera_vectors(pose))
    # The viewer's cameras are the controller's model, registered to the scan with a recorded camera error;
    # the reference cameras must sit on them with exactly that error, or the matrix is not carrying cameras correctly.
    gaps = [np.linalg.norm(np.array(c["position"]) - np.array(reference[Path(c["name"]).name]["position"])) for c in meta["cameras"]]
    rms = float(np.sqrt(np.mean(np.square(gaps))))
    print(f"{args.scene}: reference cameras against the viewer's cameras: RMS {rms * 100:.2f} cm (recorded {placement['camera_rmse_m'] * 100:.2f} cm), max {max(gaps) * 100:.2f} cm over {len(gaps)} cameras")
    if abs(rms - placement["camera_rmse_m"]) > args.tolerance:
        raise SystemExit("reference cameras do not land on the viewer's cameras as the recorded registration says; the scan placement does not carry over to cameras")

    methods = []
    for method_id, label, path, note in METHODS[args.scene]:
        source = Path(path)
        if not source.exists():
            print(f"  skipped {method_id}: {source} not found")
            continue
        poses = load_poses(source, images_dir)
        report = pose_errors(poses, ref)
        sim = report.transform
        cameras = place_cameras(poses, ref, matrix, sim)
        # The cameras written here must reproduce the metric the earlier days reported.
        recomputed = float(np.sqrt(np.mean([c["error_m"] ** 2 for c in cameras])))
        if abs(recomputed - report.ate_rmse_m) > 1e-4:
            raise SystemExit(f"{method_id}: exported camera errors give RMSE {recomputed:.5f} m, eval.metrics says {report.ate_rmse_m:.5f} m")
        trim_sim, fitted = trimmed_fit(poses, ref)
        trimmed = place_cameras(poses, ref, matrix, trim_sim, fitted)
        fit_errors = [c["error_m"] for c in trimmed if c["fitted"]]
        rest_errors = [c["error_m"] for c in trimmed if not c["fitted"]]
        methods.append({
            "id": method_id, "label": label, "note": note,
            "n_registered": len(cameras), "n_images": len(ref),
            "camera_rmse_m": round(report.ate_rmse_m, 5), "camera_median_m": round(report.ate_median_m, 5), "camera_max_m": round(report.ate_max_m, 5),
            "rotation_median_deg": round(report.rotation_err_median_deg, 3), "rotation_max_deg": round(report.rotation_err_max_deg, 3),
            "cameras": cameras,
            "trimmed": {
                "n_fitted": len(fit_errors), "fit_rmse_m": round(float(np.sqrt(np.mean(np.square(fit_errors)))), 5),
                "rest_median_m": round(float(np.median(rest_errors)), 5) if rest_errors else None, "cameras": trimmed,
            },
        })
        print(f"    best {len(fit_errors)} of {len(trimmed)}: fit RMSE {np.sqrt(np.mean(np.square(fit_errors))) * 100:.2f} cm, the rest median {np.median(rest_errors) * 100 if rest_errors else float('nan'):.2f} cm")
        print(f"  {method_id}: {len(cameras)}/{len(ref)} cameras, RMSE {report.ate_rmse_m * 100:.2f} cm, median rotation {report.rotation_err_median_deg:.2f} deg")

    (out_dir / "methods.json").write_text(json.dumps({"scene": args.scene, "reference": reference, "methods": methods}, separators=(",", ":")))
    print(f"wrote {out_dir / 'methods.json'} ({(out_dir / 'methods.json').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
