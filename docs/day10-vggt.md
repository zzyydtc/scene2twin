# Day 10: a feed-forward reconstruction model (VGGT) against structure-from-motion

Question: can a pretrained network that predicts cameras and depth in one pass (VGGT, Wang et al.,
CVPR 2025) replace feature matching, mapping and bundle adjustment? Two things were tried on the same
two scenes and scored with the same metrics as before: VGGT on its own, and VGGT's cameras as the
starting point for COLMAP.

## What was run

- **`feedforward/vggt_run.py`: VGGT alone.** All images of a scene go through VGGT-1B (the
  1-billion-parameter model, bfloat16) in one forward pass at 518 px wide (518 x 350 for these photos,
  which are 6,048 x 4,032). It returns a camera (pose and field of view) and a depth map with a
  confidence for every image. Cameras are scored against the ETH3D reference poses with the same
  Sim(3) fit as every other model here; points are the depth maps unprojected with the predicted
  cameras, thinned to 300,000 of the most confident, scored against the laser scan.
- **`feedforward/refine.py`: VGGT as the starting point for COLMAP.** The predicted poses and focal length
  are written as a COLMAP model without points. COLMAP triangulates the controller's verified SIFT
  matches with those poses and bundle-adjusts, and this repeats with tighter thresholds (13 rounds;
  the first ones accept up to 24 px reprojection error, because the starting poses are centimetres
  off). No initial pair is chosen and no image is registered one by one.
- The VGGT code is cloned to `data/third_party/vggt` and the weights (5.0 GB) are downloaded to
  `data/models/vggt-1b`; neither is in the repository. Both are covered by Meta's VGGT licence
  (`LICENSE.txt` in that clone). Nothing is trained here.

## VGGT on its own

| | `office` (26 images) | `delivery_area` (44 images) |
|---|---|---|
| Time for the forward pass (GPU) | 6.8 s | 14.3 s |
| Peak GPU memory | 9.8 GB | 10.4 GB |
| Camera RMSE against the reference | 7.0 cm | 40.4 cm |
| ... as a share of the camera spread (RMS distance from the centroid: 55.7 cm, 322.9 cm) | 12.6% | 12.5% |
| Median / largest camera rotation error | 1.33 / 3.45 deg | 3.15 / 6.86 deg |
| Same scene, incremental SfM with SIFT (Day 8) | 0.74 cm | 1.44 cm |
| Dense points, most confident 50%: median error to the scan, within 2 cm | 73 mm, 20.3% | 163 mm, 6.9% |
| Dense points, most confident 10% | 18.5 mm, 54.5% | 116 mm, 12.8% |
| Focal length x / y against the reference | 3456.6 / 3430.9 against 3437.8 / 3436.0 | 3456.6 / 3400.4 against 3408.6 / 3408.9 |

- **It is a coarse reconstruction.** The camera error is 9.5 times (`office`) and 28 times
  (`delivery_area`) the SfM error in centimetres, and the same share of the scene size in both scenes
  (about 12.5%), so it grows with the room. The median rotation errors of 1.3 and 3.2 degrees are 3 and 40 times
  SfM's 0.39 and 0.08 degrees.
- **The error is not a scale problem**: the Sim(3) fit already absorbs scale (VGGT's scale is
  arbitrary: the fit needed factors of 2.5 and 10.9). That fit uses the reference poses, as it does for
  SfM, so the numbers are not available in a real room without a reference.
- **The focal length is close to the reference** (within 0.5% and 1.4% horizontally), but the
  horizontal value is identical in the two scenes (3456.6), so I would not read it as a measurement.
- **Confidence matters a lot for the points.** Keeping the most confident tenth cuts the median error
  from 73 to 18.5 mm on `office`, but on `delivery_area` it stays at 116 mm. A threshold chosen on one
  scene does not carry to the other.
- No image is dropped: VGGT places every image, so it cannot fail by losing a group of images the way
  the learned matching of Day 9 did.

## VGGT's cameras as the starting point for COLMAP

Same SIFT database as the incremental runs. "Fixed" and "refined" are the principal point (Day 8).

| | Camera RMSE | Median rotation error | Point median error | Within 2 cm | Points | Seconds |
|---|---|---|---|---|---|---|
| **`office`**, incremental SfM, fixed | 0.74 cm | 0.39 deg | 8.4 mm | 95.3% | 9,479 | 13 |
| VGGT start, fixed | 3.07 cm | 0.67 deg | 8.9 mm | 90.3% | 7,923 | 44 |
| VGGT start, refined | 13.43 cm | 0.97 deg | 438.5 mm | 12.3% | 9,073 | 59 |
| **`delivery_area`**, incremental SfM, fixed | 1.44 cm | 0.08 deg | 15.2 mm | 58.0% | 28,961 | 65 |
| VGGT start, fixed | 1.47 cm | 0.09 deg | 15.6 mm | 57.1% | 29,136 | 102 |
| incremental SfM, refined | 0.34 cm | 0.07 deg | 10.7 mm | 70.3% | 29,074 | not timed |
| VGGT start, refined | **0.31 cm** | 0.08 deg | 10.7 mm | 69.9% | 29,113 | 105 |

Seconds count the triangulation and bundle-adjustment rounds only (not VGGT's pass, not the SIFT
features and matches), against the incremental mapper's time on the same database.

- **On `delivery_area` the bundle adjustment from VGGT's cameras ends where the incremental mapper
  does**, from a start 40 cm off: 1.47 against 1.44 cm with the principal point fixed, 0.31 against
  0.34 cm refined. The cost is time (102 s against 65 s) plus VGGT's 14 s.
- **On `office` it does not.** With the principal point fixed the camera error is 3.07 cm, and almost
  all of it is one camera: the median camera is 0.94 cm off, DSC_0239 is 14.8 cm off. With the
  principal point refined the model is wrong (13.43 cm, median point error 44 cm)
  although its mean reprojection error is 0.38 px, lower than the correct run's 0.45 px. The worst
  cameras are DSC_0238 (40 cm), DSC_0237, DSC_0241, DSC_0239 and DSC_0236: the five-image group that was
  already hard in Day 9 (linked to the rest by two weak pairs and with optical axes up to 64 degrees apart).
  The refined principal point itself (3043, 2014) is within 4 px of the reference one, so the failure is the
  group's placement, not the intrinsics.
- **This is not better than SfM anywhere here.** It matches it on one scene and is worse on the other,
  and it needs the SIFT matches, so it does not remove the matching step.

## What this shows

- A feed-forward model gives a usable first layout in seconds (7 to 14 s for 26 to 44 images, no
  matching), but its cameras are 12.5% of the scene size off, roughly 10 to 30 times the SfM error.
- As a starting point for bundle adjustment it is good enough to reach the same solution as the
  incremental mapper on `delivery_area`, a hall where COLMAP's own global mapper failed in Day 8. It
  is not a safe starting point where a group of images is weakly connected (`office`), and there a
  reprojection error of 0.38 px sat on top of a wrong model, again.
- Whether it helps where SfM fails (very sparse overlap, textureless rooms) is not tested: both scenes
  are ones where SfM works.

## Limits

- Two scenes, one camera each, VGGT-1B at its default 518 px input only, one run per setting (GPU
  inference is not guaranteed bit-identical, I did not repeat it). The photos are shrunk to 518 px wide,
  so any detail the model could use at 6,048 px is not available to it.
- The refinement rounds and thresholds were set while looking at `office` and then reused unchanged on
  `delivery_area`. Their effect on the result was not explored beyond adding more rounds on `office`
  (4 rounds gave 12.7 cm, 13 rounds 3.07 cm).
- MASt3R-SfM, the other model named in the plan, was not run: it needs a separate install and
  weights with their own licence, and VGGT already answers the question of whether a feed-forward
  front end can replace the matching, to the extent two scenes can.
- The confidence quantiles for the dense points (50% and 10%) are two values I picked, not tuned.
- The per-camera error and the camera-spread numbers come from scratch scripts in `outputs/`, not from
  the repository; `feedforward/` and `eval/compare_sfm.py` are in it.
