# Day 8: incremental against global structure-from-motion

The baseline (tag `baseline`) reconstructs with COLMAP's incremental mapper. COLMAP 4 also ships a
global mapper (the GLOMAP method: rotation averaging, then global positioning of cameras and points,
then bundle adjustment), exposed in `pycolmap` 4.2.1 as `global_mapping`. This compares the two on the
same input. Code: `eval/compare_sfm.py`; tests: `tests/test_compare_sfm.py`.

Scenes: `office` (26 photos) and `delivery_area` (44 photos), each with the OPENCV model the
controller chose. Both engines read a copy of the controller's `database.db`, so the features and the
verified matches are identical and only the mapping differs. Each run keeps the model with the most
registered images and is scored like the baseline: camera centres against the reference poses after a
Sim(3) fit, and the points against the laser scan.

Rerunning the incremental mapper on the stored database reproduces the baseline numbers exactly
(`office` 0.74 cm and 9,479 points, `delivery_area` 1.44 cm), so the comparison starts from the same
reconstructions as everything else in this repository.

## Result with COLMAP's defaults

Mean over 5 seeds (per-seed values are in the local `results.json` that `eval.compare_sfm` writes).

| | `office` incremental | `office` global | `delivery_area` incremental | `delivery_area` global |
|---|---|---|---|---|
| Images registered | 26 | 26 | 44 | 44 |
| Points | 9,479 | 9,396 | 28,961 | 27,470 |
| Camera RMSE | 0.74 cm | 0.73 cm | 1.44 cm | **283 cm** |
| Median camera rotation error | 0.39 deg | 0.40 deg | 0.09 deg | **175 deg** |
| Point median error to the scan | 8.4 mm | 8.3 mm | 15.2 mm | **1,054 mm** |
| Points within 2 cm of the scan | 95.3% | 95.7% | 58.0% | **0.7%** |
| Mean reprojection error | 0.415 px | 0.412 px | 0.560 px | 0.583 px |
| Time | 13.7 s | 7.5 s | 66 s | 70 s |

- **On `office` the two engines give the same reconstruction** to within the noise of the metrics:
  camera RMSE 0.74 and 0.73 cm, points 8.4 and 8.3 mm from the scan. The global mapper is about 1.8
  times faster (7.5 s against 13.7 s); at 26 photos that is a few seconds.
- **On `delivery_area` the global mapper fails with its defaults, in every seed,** and the failure is
  not visible in the numbers COLMAP reports: all 44 images are registered and the mean reprojection
  error (0.58 px) is as good as the incremental one (0.56 px).

The seeds are not independent replicates. The incremental mapper returned the identical model for every
seed on both scenes (standard deviation 0), and the global mapper's results moved by at most four points
and 0.5 cm in 283 cm. The spread above says the engines are repeatable on this input, not that the
result would hold on other scenes.

## What the `delivery_area` failure is

The global model is two correct halves assembled wrongly:

- Splitting the 44 cameras by whether their relative rotations agree with the reference gives two
  groups, 23 and 21 cameras. Each group on its own fits the reference poses to 1.6 cm and 1.0 cm RMS,
  with the same scale (1.85 and 1.83), so each half is a good reconstruction.
- The two halves differ from each other by a rotation of 167 degrees about the vertical axis. That
  is what gives the 175 degree median rotation error and the 283 cm camera error.
- This is invisible to the reprojection error because most tracks lie inside one half.

The two halves are not weakly connected: the database holds 84 verified pairs between them, 16 with more
than 100 inliers (up to 2,216). A rotation-averaging solution that is wrong for one half must therefore
have been pulled by pairs that are themselves wrong, and the weak pairs are the likely source:

- Using the stored essential matrices of the calibrated pairs (205 of 385) and the reference poses, the
  fraction of pairs whose relative rotation is more than 5 degrees from the truth falls with the inlier
  count: 67% for 15 to 29 inliers, 50% for 30 to 49, 25% for 50 to 99, 13% for 100 and more. The
  weak bins hold 9 and 12 pairs, so this is a trend, not a measurement. The other 175 pairs use an
  uncalibrated geometry for which the stored matrix is not a usable check, so they are left out.
- The cross-group pairs include 36 with 15 to 29 inliers, more than the 16 strong ones.

Raising the minimum number of verified matches per pair removes those pairs. Options tried on
`delivery_area`, one seed each, chosen after seeing the failure:

| Change to the global mapper | Result |
|---|---|
| none (minimum 15 matches) | 283 cm, 44 images |
| stratified rotation averaging off | identical to the default |
| rotation weight HALF_NORM | two models of 23 images, 1.6 cm for the larger |
| relative pose not decomposed, or no random initial positions | no model |
| minimum 50 matches | **2.3 cm, 44 images** |

So the two halves exist as separate models (the HALF_NORM row) and the default joins them wrongly.

## With a higher match threshold

The minimum number of verified matches per pair (COLMAP's default is 15) was then set to 30, 50 and 100
for both engines, one seed each. Camera RMSE in cm (images registered when not all):

| Minimum matches | `office` incremental | `office` global | `delivery_area` incremental | `delivery_area` global |
|---|---|---|---|---|
| 15 (default) | 0.74 | 0.73 | 1.44 | 283 |
| 30 | 0.74 | 0.74 | 1.45 | 2.29 |
| 50 | 0.72 | 0.73 | 1.47 | 2.29 |
| 100 | 0.46 (19 of 26) | 0.69 (25 of 26) | 1.46 | 2.30 |

- **On `delivery_area` any threshold from 30 up repairs the global mapper**, to 2.3 cm. That is still
  58% worse than the incremental mapper's 1.45 cm, and the points are worse too (median 18.2 mm against
  15.1 mm; 52.7% within 2 cm against 58.0%), with about 5% fewer points. The incremental mapper takes 66 to 72 s and the global mapper 36 to 40 s,
  1.8 to 1.9 times faster.
- **On `office` a threshold of 100 costs images** in both engines, so a larger threshold is not free.
  The incremental mapper loses 7 of 26 and its camera RMSE of 0.46 cm is over fewer, easier cameras
  and is not comparable.
- Thresholds of 30 and 50 leave `office` unchanged for both engines.

## Repairing the global mapper without a threshold

A fixed match threshold removes weak pairs, but the weak pairs are not the whole story and a threshold
costs images when set high (`office` at 100). Two checks first, then a filter that uses no reference.

**The cause, tested directly.** Using the reference poses and the reference camera calibration, each
verified pair was scored by how many of its inlier matches lie within 3 px of the epipolar line the
reference predicts. Of 385 pairs, 69 fail (fewer than half within 3 px): 72% of the pairs with 15 to 29
inliers (43 of 60), 65% with 30 to 49 (26 of 40), and none of the 285 pairs with 50 or more. Removing
all 69 before global mapping, with the default threshold of 15 and three seeds, gives 2.29 to 2.30 cm
with all 44 images registered, against 283 cm with none removed. (This uses the reference, so it is a
diagnosis, not a method.)

*Correction (Day 9).* An earlier version of this section scored the pairs with the camera that COLMAP
estimates, whose principal point is fixed at the image centre. The reference principal point is 13 px
from the centre on this scene, and that error alone labelled 42 fine pairs as wrong (36 of them strong ones) and missed 1; it
also gave a different removal table (including "only the strong wrong pairs: 280 cm", which tested pairs
that are in fact fine). Those numbers are withdrawn. The conclusion that weak wrong pairs cause the
failure stands, and is now cleaner: every wrong pair is weak.

**A filter that needs no reference** (`sfm/view_graph.py`, tests in `tests/test_view_graph.py`). Each
pair's relative rotation is estimated from its inlier matches with RANSAC and the focal length COLMAP
started from. For two images paired with a third, the three rotations around the loop must multiply
to the identity; a pair is dropped when fewer than half of the loops through it close within 10
degrees. On `delivery_area` this drops about 100 of 385 pairs (102 in the run checked against the
labels; the RANSAC is random). Judged by the corrected reference-based labels above (the filter never
sees them), it removes 68 of the 69 wrong pairs and also 34 pairs that were fine. On `office` it drops
22 pairs and costs no image. Its 2.11 cm on `delivery_area` (table below) is about what removing exactly
the wrong pairs gives (2.29 cm), so the remaining gap to the incremental mapper is the global mapper
itself, not leftover bad pairs.

`global_filtered` in `eval.compare_sfm` is this filter, then the global mapper, then one more bundle
adjustment (the incremental mapper ends the same way). Three seeds each, default minimum of 15 matches:

| | `office` incremental | `office` global | `office` filtered | `delivery_area` incremental | `delivery_area` global | `delivery_area` filtered |
|---|---|---|---|---|---|---|
| Images registered | 26 | 26 | 26 | 44 | 44 | 44 |
| Camera RMSE (cm) | 0.74 | 0.73 | 0.72 (0.71 to 0.73) | 1.44 | 283 | **2.11 (1.75 to 2.29)** |
| Point median error (mm) | 8.5 | 8.3 | 8.4 | 15.2 | 1002 | 18.2 |
| Points within 2 cm | 95.3% | 95.7% | 95.9% | 58.0% | 0.9% | 52.7% |
| Time (s) | 13.0 | 7.8 | 22.0 | 64.5 | 67.3 | 73.7 |

- **The default failure is gone**: all three seeds register 44 images and none flips. It is one call
  before global mapping and has no scene-specific number, though the 0.5 support and the 10 degree
  tolerance were chosen after looking at this scene. Only one other setting was tried end to end
  (support under 0.3, which dropped 79 pairs and gave 2.3 cm); other tolerances were not.
- **It is not as accurate as the incremental mapper** on `delivery_area`: 2.11 cm against 1.44 cm,
  and the result varies by seed (1.75 to 2.29 cm), because the global positioning starts from random
  positions. The incremental mapper returned the same model every time.
- **It is not faster.** The pairwise RANSAC costs about 14 s on `office` and 40 s on `delivery_area`,
  which is more than the global mapper saves; filtered global took 22 s against the incremental
  mapper's 13 s on `office`, and 74 s against 65 s on `delivery_area`. Estimating the rotations
  with fewer RANSAC iterations was not tried.

## A larger effect than the engine: the principal point

The baseline's bundle adjustments keep the principal point fixed at the image centre. Letting
every engine refine it (`--refine-principal-point`; for `global_filtered` this includes the last
bundle adjustment) changes much more than the choice of engine does:

| Camera RMSE (cm), 3 seeds | `office` incremental | `office` global | `office` filtered | `delivery_area` incremental | `delivery_area` global | `delivery_area` filtered |
|---|---|---|---|---|---|---|
| Principal point fixed (baseline) | 0.74 | 0.73 | 0.72 | 1.44 | 283 | 2.11 |
| Principal point refined | **0.45** | 0.46 | **0.43** | **0.33** | 283 | 0.56 (0.48 to 0.65) |

- Refining the principal point cuts the incremental mapper's camera error by 39% on `office` and by
  77% on `delivery_area` (point median error from 15.2 to 10.8 mm there). The
  baseline pipeline did not do this and is unchanged here. It is not a gain on every metric: on `office`
  the median point error to the scan got worse, from 8.5 to 10.6 mm (the points within 2 cm stayed at 95%).
- With it, the filtered global mapper is within 0.25 cm of the incremental one on `delivery_area` (0.56
  against 0.33 cm) and equal on `office` (0.43 against 0.45 cm). The unfiltered global mapper still
  fails on `delivery_area` (283 cm), so the filter is what repairs it.
- **The refined principal point is the real one.** The ETH3D calibration puts the principal point at
  (3040.2, 2010.2) on `office` and (3036.9, 2012.7) on `delivery_area`, 17 px and 13 px from the image
  centre (3024, 2016) that the baseline fixes. The refined values land within 4 px (3043.9, 2013.6) and
  1 px (3037.8, 2012.0) of the reference. So the gain is not the refinement absorbing some other
  error: the baseline fixed a value that was wrong by 13 to 17 px. (Two scenes, one camera each.)

## Trying it in the controller

`agent.controller --refine-principal-point` (off by default, so the baseline is unchanged) passes the
option through `agent.tools.reconstruct` to all three candidate camera models. Both scenes were rerun
end to end with the same tapes as before. Single runs; the incremental mapper is deterministic here.

| | `office` fixed | `office` refined | `delivery_area` fixed | `delivery_area` refined |
|---|---|---|---|---|
| Camera model chosen | OPENCV | OPENCV | OPENCV | OPENCV |
| Camera RMSE | 0.74 cm | **0.43 cm** | 1.44 cm | **0.34 cm** |
| Median camera rotation error | 0.39 deg | 0.08 deg | 0.08 deg | 0.07 deg |
| Median point error to the scan | 8.5 mm | 10.9 mm | 15.2 mm | **10.7 mm** |
| Points within 2 cm of the scan | 95.3% | 95.2% | 58.0% | **70.3%** |
| Structure label agreement with the scan | 84.8% | 87.8% | 62.0% | 62.0% |
| Metric scale from the tape | +3.0% off | +2.0% off | +0.3% off | **none** |

- **The camera-model choice does not change**: OPENCV is still chosen over SIMPLE_RADIAL (reprojection
  55% to 59% better) and the fisheye model is still rejected (6% and 9% gain, 20% needed).
- **The pose gain is real on both scenes**, and the points move closer to the scan on `delivery_area`;
  on `office` the points get slightly worse, so the refined principal point is not better by every
  measure.
- **It changed a downstream decision on `delivery_area`, for the worse.** The floor-plausibility gate
  went from `warn` to `pass` (the floor plane now spans the room). The controller's rule for
  disagreeing tapes drops the height tape only when the floor gate warns, so with a pass it could not
  single out a tape (scales disagree by 12.0%) and produced no metric scale, where the fixed
  run kept the horizontal tape and was 0.3% off. The rule behaved as designed; the height tape was
  still the wrong one, and this is the gate that failed validation earlier (it gave the same value for a wrong floor and a
  correct one), so which way it falls is close to arbitrary.
- **The structure step is sensitive.** On `office` the wall agreement went from 40% to 94% and overall
  agreement from 84.8% to 87.8% with no change to the structure code; I did not trace which wall
  plane changed. `delivery_area` was unchanged (62.0%).
- A second, small model appeared in the `office` OPENCV and fisheye runs with a refined principal point
  ("2 models produced"); the largest, with all 26 images, was used.

So the option is kept opt-in. Making it the default would improve the poses on both scenes but changes
what the gates decide on `delivery_area`, and two scenes are not enough to settle it.

## What this shows

- **The global mapper's failure on `delivery_area` was caused by weak, wrong pairs, and a
  reference-free loop-closure filter repairs it** (283 cm to 2.1 cm, or 0.56 cm with the principal
  point refined), without a tuned threshold and without costing images on `office`.
- **No accuracy gain from the global mapper on these scenes.** Where it works it matches the incremental
  mapper on `office` and is about 20% to 70% worse on `delivery_area` (0.56 against 0.33 cm; 2.11 against
  1.44 cm). Once the filter is paid for, it is not faster either. The advantage of global methods
  grows with the number of images; these scenes have 26 and 44, so this comparison cannot show it.
- **The default global mapper is less safe on a hall with repeated structure.** Its failure produced no
  warning in the reconstruction's own statistics (44 images, 0.58 px). Only the comparison with an
  independent reference, or a check that the camera graph is rotation-consistent, shows it. The
  incremental mapper succeeded on both scenes with the defaults. As in Day 2, the reprojection
  error alone does not tell a good model from a wrong one.
- **The principal point matters more than the engine.** The baseline fixes it 13 to 17 px from the
  reference; the refined value is within 4 px. It is available in the controller as an opt-in option;
  it improves poses on both scenes and changes one gate decision, so it is not the default yet.

## Limits

- Two scenes, one camera each, one machine (a laptop CPU; this `pycolmap` build has no CUDA, so
  the GPU options fell back to the CPU). Times are wall clock of single runs and include no warm-up.
- Only COLMAP's own two mappers are compared, on COLMAP's own SIFT features and matches. A different
  feature matcher or a learned reconstruction model (for example VGGT or MASt3R-SfM) is not tested here.
- The threshold sweep, the option probe, the filter parameters and the diagnosis were all done on
  `delivery_area`, after the failure was seen; `office` is the only scene on which the filter was checked
  not to do harm. Whether it repairs other scenes is not tested.
- The seeds of the incremental mapper are not independent (it returns the same model); the global
  mapper's seeds are, and its spread is reported where it matters.
- The controller's default is unchanged: the incremental mapper with a fixed principal point.
