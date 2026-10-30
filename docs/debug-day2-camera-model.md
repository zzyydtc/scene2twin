# Debug log: sparse points 8 cm off the laser scan (ETH3D `office`)

Scene: ETH3D high-res multi-view `office` (26 DSLR images, 6048x4032, laser-scan ground truth).
Tools: pycolmap 4.2.1 (SfM), Open3D (ICP, voxel filter), own code in `eval/` and `registration/`.

## 1. Setup

| Item | Detail |
|---|---|
| Images | `office_dslr_jpg.7z`, 26 images, plus reference calibration in COLMAP text format |
| Ground truth | `office_scan_clean.7z`: `scan1.ply`, `scan2.ply` (about 24.1 M points each) and `scan_alignment.mlp` (one 4x4 transform per scan) |
| Reference scan used | both scans transformed by the `.mlp` matrices, merged, voxel-downsampled to 2 cm: 419,482 points |
| Reference camera | one `THIN_PRISM_FISHEYE` camera (12 distortion parameters), f about 3438 px |
| Metric frame | SfM output has an arbitrary frame and scale, so a Sim(3) is fitted before measuring errors |

Evaluation code (all written in this repo):
- `registration/sim3.py`: Umeyama closed-form Sim(3), with a reflection guard.
- `eval/metrics.py`: camera-position error (ATE after Sim(3) on camera centres), orientation error, point-to-scan distance (accuracy), scan-to-point distance (completeness), reprojection statistics.
- `eval/colmap_io.py`: reads reference/estimated poses, applies the `.mlp` transforms.
- `tests/test_eval.py`: 8 tests (known-transform recovery, planar degeneracy, perturbation detection, thresholds, MeshLab parsing).

## 2. Symptom

Baseline: COLMAP default camera model `SIMPLE_RADIAL`, automatic camera mode.

| Metric | Value |
|---|---|
| Registered images | 26 of 26, one model, 9,277 points, 0.7 min |
| Mean reprojection error | 0.99 px (median 0.90, p95 1.98, max 3.74); 92% of tracks have length >= 3 |
| Camera-position error (ATE RMSE) | 2.3 cm (median 1.4 cm, max 6.6 cm) |
| Camera orientation error | median 0.58 deg, max 1.22 deg |
| Point-to-scan distance | mean 7.0 cm, **median 8.1 cm** |
| Points within 2 / 5 / 10 cm of the scan | 9% / 23% / 98% |

Cameras were accurate to about 2 cm while points were 8 cm off, with a reprojection error that
would normally pass a sanity check. The reference scan has a point spacing of about 1.5 cm after
filtering (estimated from point count and room size), so a median of 8 cm is a systematic
offset, not noise.

## 3. Hypotheses

1. The scan transform (`.mlp`) is applied incorrectly, so the ground truth is misaligned.
2. The camera-centre-only Sim(3) is a poor alignment for the points.
3. The camera model is wrong: the lens is a fisheye, the baseline used a single-radial-term model.

## 4. Experiments

### 4.1 Reference points against the scan (tests hypothesis 1)

The reference `points3D.txt` (3,461 points) is in the same frame as the scan by construction.
- With the `.mlp` transforms applied: median 0.98 cm, 75% within 2 cm, 87% within 5 cm.
- With `scan1.ply` untransformed: median 12.8 cm.

Conclusion: the scan alignment is correct. The problem is in our reconstruction.

### 4.2 Point-based ICP with scale on the baseline (tests hypothesis 2)

Scaled point-to-point ICP (max correspondence distance 0.15 m) from the Sim(3) result.
- Median point-to-scan distance: 8.1 cm -> 1.0 cm; 86% of points within 2 cm.
- The ICP changed the scale by 2.4%.

Interpretation: the structure is internally consistent enough that a point-based fit recovers it,
but the camera-centre Sim(3) has the wrong scale for the points. A 2.4% scale error on a room
roughly 3 to 4 m across is of the order of the observed 8 cm (an estimate, not measured).
This is a symptom of distorted structure, not a root cause.

### 4.3 Camera model change (tests hypothesis 3)

Re-ran SfM with `OPENCV_FISHEYE` and a single shared camera.

| Run | Camera model / mode | ATE RMSE | Point median to scan | Within 2 / 5 / 10 cm | Mean reproj. |
|---|---|---|---|---|---|
| Baseline | SIMPLE_RADIAL, auto | 2.3 cm | 8.1 cm | 9% / 23% / 98% | 0.99 px |
| Control | SIMPLE_RADIAL, single camera | 2.5 cm | 10.5 cm | 10% / 21% / 38% | 0.98 px |
| Fix | OPENCV_FISHEYE, single camera | 1.1 cm | **1.35 cm** | 82% / 98% / 99% | 0.45 px |

The control run (same camera mode, original model) isolates the cause: the camera model, not the
camera mode, produces the improvement. The baseline and control differ by about 2 cm in point
median, so the bad configuration itself varies by that much between setups; the fix is far
outside that range. Scale values are not comparable between runs because SfM scale is arbitrary.

## 5. Mistakes along the way (kept for the record)

- The fisheye run split the scene into **two models** (7 images and 26 images). The evaluation
  initially read `sparse/0`, the 7-image fragment, and reported a misleading 24 cm point error.
  The 26-image model was `sparse/1`. `scripts/day1_baseline.py` now writes the largest model to `best/`.
- Passing `camera_model` directly to `pycolmap.extract_features` raised a `TypeError`; in pycolmap 4.2.1
  it goes through `ImageReaderOptions`.
- A scripted patch to a diagnostic left a half-disabled code block that referenced an undefined
  variable; it was rewritten rather than patched again.
- Reconstructions are not bit-reproducible: two runs of the same configuration gave 9,458 and 9,484 points
  (and 7,145 and 7,162 for the fragment). Differences of about 0.3% in point count are noise.

## 5b. Addendum (Day 5): the fisheye model was not the best fix

Day 5's controller compared three camera models and preferred `OPENCV` (pinhole with radial and
tangential distortion) over `OPENCV_FISHEYE`. Evaluated against the scan afterwards:

| Camera model | ATE RMSE | Point median to scan | Within 2 / 5 cm | Mean reproj. |
|---|---|---|---|---|
| SIMPLE_RADIAL | 2.5 cm | 10.5 cm | 10 / 21% | 0.98 px |
| OPENCV_FISHEYE | 1.1 cm | 1.35 cm | 82 / 98% | 0.45 px |
| OPENCV | 0.74 cm | 0.84 cm | 95 / 98% | 0.42 px |

So the cause stands (an inadequate distortion model), but the claim above that the *fisheye*
model was the remedy is too narrow: the lens is a fisheye-type camera in the reference
calibration, yet a richer pinhole-plus-distortion model fits this data better. The practical
lesson is to compare several models, which is what the agent does.

## 6. Conclusion

Lens-model mismatch distorted the reconstructed structure while the reprojection error stayed close to
1 px. A reprojection error below 1 px is a common acceptance rule of thumb, and it did not flag this.
Comparing against independent reference geometry did.

## 7. Limits of this analysis

- One scene and one lens; no claim that the same gap appears elsewhere.
- I did not separate which part of the fisheye model matters (radial terms versus the tangential /
  prism terms of the true `THIN_PRISM_FISHEYE`).
- The remaining 1.35 cm median error has not been broken down into calibration, matching and
  scan noise.
- Point accuracy here is measured on sparse points only; completeness needs a dense reconstruction.

## 8. Consequences for the agent layer

- Quality gates must check independent geometry (reference scan, tape measurements, ICP residual
  and its implied scale change), not only reprojection error.
- When reprojection error is acceptable but the scale correction needed by point-based fitting is
  larger than a threshold (here 2.4%), the agent should try alternative camera models.
- When mapping returns several models, the agent must choose the largest and record the choice.
- The agent should record run-to-run variation so a small change is not mistaken for an improvement.
