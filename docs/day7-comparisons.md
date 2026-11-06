# Day 7: two comparisons against the baseline

The tag `baseline` (commit `344a818`) is the project before this work: COLMAP sparse SfM, room-axes
structure, SAM lifting, tape scale, evaluation against the ETH3D laser scans. Both comparisons below
add a method from photogrammetry or remote sensing and score it with the same data and metrics as the
baseline. Scenes: `office` (26 photos) and `delivery_area` (44 photos), the OPENCV model the controller
chose for each (the registration comparison also uses the Day 3 fisheye model of `office`). Limits that applied before still apply: two scenes, one camera each.

## 1. Registration to the scan: correlating occupancy grids

Day 3 left registration unsolved: the room-axes method found the rotation only approximately and not translation or
scale, and the cause was suspected to be partial coverage (the cameras see less of the room than the
scan does) and never tested.

Method (`registration/raster.py`, tests in `tests/test_raster.py`): both clouds are rasterised into
occupancy grids (0.15 m voxels, dilated by one voxel). For each of the 24 axis-aligned rotations
between the two room-axes frames and each scale on a grid, an FFT cross-correlation scores every
translation at once. The score is the cosine similarity between the two grids *inside the box the
reconstruction covers*, so scan structure the cameras never saw is not counted against a match.
The best six rotations are refined with point-to-plane ICP with scale, over match distances
0.4, 0.25, 0.15, 0.08 and 0.04 m, and ranked by occupancy IoU as before. A measured scale
(`--scale-prior`) narrows the scale grid; without one the grid spans a factor of four.

| Scene and model | Method | Camera RMSE | Point median | Within 2 cm | Scale | Time |
|---|---|---|---|---|---|---|
| `office`, OPENCV | room axes (baseline) | 107.5 cm | 1.09 cm | 63% | 0.1696 | 319 s |
| | raster, scale searched | **0.90 cm** | 0.79 cm | 96% | 0.1384 | 74 s |
| | raster, tape scale prior | **0.90 cm** | 0.79 cm | 96% | 0.1384 | 42 s |
| | fit to the reference poses (not deployable) | 0.74 cm | 0.84 cm | 95% | 0.1385 | |
| `office`, fisheye model of Day 3 | room axes (baseline) | 106.3 cm | 1.23 cm | 63% | 0.1867 | 204 s |
| | raster, scale searched | **1.98 cm** | 0.83 cm | 95% | 0.1523 | 65 s |
| | fit to the reference poses | 1.10 cm | 1.35 cm | 82% | 0.1534 | |
| `delivery_area`, OPENCV | room axes (baseline) | 964 cm | 63.5 cm | 4% | 0.9509 | 1,272 s |
| | raster, tape scale prior | **1.98 cm** | 1.18 cm | 69% | 0.8440 | 5,566 s |
| | fit to the reference poses | 1.44 cm | 1.52 cm | 58% | 0.8432 | |

"Camera RMSE" is the distance between the estimated and reference camera centres after the
registration; one random seed per row. The scan is the 2 cm reference throughout the scoring;
`delivery_area` registers to a 0.1 m copy of it (236,473 of 5,152,797 points), because the
baseline's ICP did not finish on the full scan in over five hours.

What this shows:

- **The Day 3 suspicion holds: partial coverage was the main obstacle.** The same reconstructions,
  which the room-axes method placed 1 m (and on `delivery_area` 10 m) away, register to within
  about 1 to 2 cm of the reference fit once the correlation ignores the part of the scan the
  cameras did not see.
- **The scale is found, not only checked.** Without a prior the scale lands within 0.1% of the
  reference fit on `office` (0.1384 against 0.1385), better than the tape value used as the prior
  (0.1427 against 0.1385). The prior made the search 1.8 times faster and changed nothing else.
  On `delivery_area` only the prior run was done, so scale-free recovery is not shown there.
- **`delivery_area` is slow.** The correlation grid grows with the scene (70 m by 48 m), and the
  run took 93 minutes, four times the baseline's time. This is a limit for large scenes; no speed
  work was done.

What did not work first, kept because it changed the method:

- The first version used point-to-point ICP, as the baseline does. On `office` it left a camera
  RMSE of 23.6 cm (27.2 cm without the prior), with all 26 cameras 12 to 30 cm off and the
  vertical axis exact. The cause was a rotation 8.2 degrees from the true one: the room axes
  estimated from sparse points are good to a few degrees, and point-to-point ICP could not remove
  the rest because each point is pulled to its nearest scan point rather than its wall plane. The
  true pose scored better than the found one on IoU, nearest-neighbour distance and fraction
  within 2 cm, so the objective was not at fault; the optimiser was.
- A second correlation pass at 4 cm and 2 cm voxels inside a 30 cm window around the coarse pose
  did not help (camera RMSE stayed at 20 to 22 cm), because the rotation, not the voxel size, was
  wrong. It was removed.
- Point-to-plane ICP (`icp_plane_sim3`) fixed it: from the same starting pose, 23.6 cm became
  1.2 cm. I chose it after looking at the `office` result against the reference, so `office` is a
  tuned result; the fisheye model and `delivery_area` were run afterwards with no further changes
  and are the independent checks.

Limits: three runs on two scenes, one seed each. Both scenes have a usable room-axes frame (box-like
rooms), which the method needs to propose rotations; a scene without one is not tested.

## 2. Predicted precision versus observed error

Photogrammetry states a precision for every triangulated point from the geometry of its rays. Remote
sensing validation asks whether a stated uncertainty matches the error seen against independent data.
This comparison checks that for the baseline's sparse points. The baseline's own per-point quality
number, COLMAP's reprojection error, is one of the predictors compared. Code: `eval/calibration.py`,
`eval/run_calibration.py`; tests: `tests/test_calibration.py`.

Method:

- **Observed error:** point-to-plane distance from each point to the laser scan (normals from 6 cm
  neighbourhoods), after the Sim(3) fit of the camera centres to the reference poses, the same fit
  `eval.run_eval` uses. Points more than 10 cm from any scan point are left out (1% of the points in
  `office`, 6% in `delivery_area`).
- **Predicted precision:** the linearised triangulation covariance with the cameras held fixed. A ray of
  length `r` seen at focal length `f` pixels has lateral standard deviation `sigma_px * r / f`; each
  point's information matrix is the sum of `(f / (sigma_px * r))^2 * (I - d d^T)` over its rays. The
  image noise `sigma_px` is the RMS of COLMAP's per-point reprojection errors (0.55 px and 0.70 px). The
  covariance is projected onto the scan normal so it is compared with the quantity that was measured.
- **Scores:** Spearman rank correlation with the observed error; the area under the sparsification
  error curve (AUSE: remove points worst-first by a predictor and compare the RMS of the rest with the
  best possible ordering, lower is better, `random` is the reference); and coverage, the fraction of
  points whose error falls within 1, 2 and 3 predicted sigmas (a Gaussian gives 68%, 95% and 99.7%).

| | `office` | `delivery_area` |
|---|---|---|
| Points compared | 9,395 | 27,182 |
| Observed error, median / RMS / 95th percentile | 2.5 / 7.7 / 11.3 mm | 10.7 / 25.7 / 57.9 mm |
| Predicted sigma, median / RMS | 1.9 / 2.9 mm | 4.9 / 16.4 mm |
| Coverage within 1 / 2 / 3 predicted sigma | 40% / 68% / 84% | 25% / 45% / 58% |
| Same after one global factor (2.7 and 1.6) | 80% / 94% / 96% | 37% / 59% / 72% |

Ranking observed error (Spearman, and AUSE in mm; `random` is 5.9 and 17.2):

| Predictor | `office` rho | `office` AUSE | `delivery_area` rho | `delivery_area` AUSE |
|---|---|---|---|---|
| Predicted sigma along the normal | 0.22 | 4.6 | 0.44 | 7.9 |
| 1 / largest ray angle | 0.29 | **3.3** | 0.34 | 9.2 |
| Distance to the cameras | 0.12 | 4.3 | 0.54 | **5.4** |
| COLMAP reprojection error | 0.14 | 3.9 | 0.09 | 16.2 |
| 1 / track length | 0.03 | 5.0 | -0.07 | 18.9 |

Figures: [`office`](figures/office_calibration.png), [`delivery_area`](figures/delivery_area_calibration.png).

What this shows:

- **The predicted precision is too optimistic, and the errors have heavy tails.** Observed errors are
  1.3 to 2.2 times the predicted sigma at the median and far larger at the tail: 84% and 58% of points
  fall within three sigmas, where a Gaussian gives 99.7%. The covariance only includes triangulation
  noise. Pose and calibration errors and the scan's own noise are not in it, and in `delivery_area` the
  reliability curve sits about 9 mm above the diagonal even for the points predicted to be best, which
  is that systematic floor (the camera-centre fit itself leaves 14 mm RMS there).
- **COLMAP's reprojection error is a weak ranking of real error.** On `delivery_area` it is no better
  than a random order (AUSE 16.2 mm against 17.2 mm), and track length is no better. A likely reason,
  which I have not isolated, is that a poorly conditioned triangulation can fit its few observations
  well and still be placed badly along the rays.
- **Geometry ranks better, but no single predictor wins on both scenes.** The ray angle is best on
  `office`; distance to the cameras and the predicted sigma are best on `delivery_area`, where depths
  vary more. The predicted sigma is not the best on either, which suggests that the part of the error
  it models is not the part that dominates.

Limits: two scenes and one model each; the error includes the fit error of the Sim(3) used to bring the
model into the scan frame; the scan's own noise (a few millimetres) is not subtracted; `sigma_px` is
estimated from the same data it is used on. Nothing here changes the pipeline: these predictors are
not used in the labels, confidences or gates yet.

## In the viewer

Both comparisons can be inspected in the [demo](https://zzyydtc.github.io/scene2twin/viewer/) (data
exported with `scripts/export_viewer.py --scene ... --placement ...`):

- **Colour by Predicted precision or Observed error** (same scale on both), best used with Compare:
  the left side shows the sigma the geometry predicts, the right side the distance to the scan.
  Points off the scan are grey. Clicking a point shows both numbers.
- **Laser scan layer**, thinned to at most 100,000 points around the reconstruction, with a switch
  for which registration aligns it to the points. The scan moves, the points and cameras do not, so
  the photo overlay and picking are unaffected; the relative placement is the same either way. The
  panel shows each method's camera error (office 0.9 cm against 1.08 m, `delivery_area` 2.0 cm against
  9.64 m).
- The run time shown for the `delivery_area` raster result (93 min) is from the first run. The rerun that
  saved the transform for the viewer gave the same result but a wall-clock figure of 8.7 hours, most
  likely because the computer was asleep for part of it; that figure is not used.
