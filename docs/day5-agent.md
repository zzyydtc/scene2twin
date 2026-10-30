# Day 5: self-checking controller (ETH3D `office`)

Code: `agent/gates.py` (scan-free quality gates), `agent/tools.py` (reconstruction tool),
`agent/controller.py` (the loop and the decision log). Tests: `tests/test_gates.py`
(40 tests in total pass).

**What this is, and is not.** The controller is deterministic: every decision is a rule over
gate results and is written to a decision log. No language model is involved, so "agent" here
means a self-checking pipeline controller, not an LLM planner.

## Loop

1. Reconstruct with each candidate camera model (`SIMPLE_RADIAL`, `OPENCV`, `OPENCV_FISHEYE`).
2. Registration gate per model (at least 90% of images in one model).
3. Choose the model **by comparison**: from the simplest, a richer model replaces the current one
   only if reprojection error drops by at least 20% and plane confidence does not drop by more
   than 0.02. An absolute reprojection threshold is not used because a wrong model scored 0.98 px.
4. Floor gate: share of occupied floor-area cells that contain floor points (warn below 0.5).
5. Metric scale from tape measurements, cross-checked: scales must agree within 5%. If they
   disagree and the floor gate warns, the `height` tape is dropped as the likely outlier.
6. Instance gate: SAM instances spanning more than half the room in two or more axes are excluded
   as chained merges. Labels from a different reconstruction are refused.
7. Scene graph with units, and a JSON report with the full decision log.

## Run on `office` (tapes simulated from the scan: height 4.05 m, wall to wall 3.13 m)

| Decision | What the controller did | Check against ground truth (evaluation only) |
|---|---|---|
| Camera model | Chose `OPENCV` (reprojection 0.415 px vs. 0.984 px for `SIMPLE_RADIAL`; `OPENCV_FISHEYE` at 0.448 px did not beat it by 20%) | `OPENCV` is the best of the three: ATE 0.74 cm, point median 0.84 cm, 95% of points within 2 cm |
| Floor | Warned: floor points cover 47% of the occupied floor area | The "floor" is a surface above the real floor (Day 4) |
| Tapes | Scales disagreed by 19.2%; dropped `height`; used `horizontal` alone (0.1427 m per unit), flagged as not cross-checkable | True scale 0.1385: the dropped height tape would have been 22.7% off, the kept tape is 3.0% off |
| Instances | Excluded 1 chained instance (id 18) from a fresh SAM run on the chosen model | Consistent with the Day 4b finding of chained merges |
| Stale labels | Refused SAM labels computed on a different reconstruction | Caught a real mistake made during this run |

## Caveats

- **One scene.** Every threshold (20% reprojection gain, 0.5 floor coverage, 5% tape tolerance,
  50% instance extent) was set by me after seeing this scene. The floor gate in particular is
  borderline: coverage was 0.47 for `OPENCV` and 0.30 for `OPENCV_FISHEYE`, so a threshold
  of 0.5 separates them from "real floor" only by luck until it is tested on other scenes.
- **A 3% scale error remains** with a single horizontal tape; a second horizontal tape would allow
  a cross-check, but only one wall pair is observed in this reconstruction.
- **Reprojection comparison favours richer models** by construction; the 20% margin and the plane
  confidence condition are guards, not guarantees.
- **Room extent** in the scene graph comes from percentiles of all points, not wall-to-wall
  distances, and is not a measurement.

## Bugs found by running the controller

- Dimension names such as `horizontal_1` depended on an arbitrary axis order that changes between
  runs; they are now `height`, `horizontal`, or `horizontal_long` / `horizontal_short`.
- The scene graph reported a 10 m room extent because raw min/max included far outliers; it now
  uses the 0.5 and 99.5 percentiles.
- Instance labels from one reconstruction were applied to another with a different point count,
  which crashed the gate; the controller now checks and skips with a log entry.

## Second scene: ETH3D `delivery_area` (44 images, a hall about 28 x 28 x 4.6 m)

All thresholds were frozen before this run; none was changed afterwards. Tapes were again
simulated from the scan (height 4.59 m, wall to wall 28.0 m). Reference accuracy comes from the
same scan-based evaluation as before.

| Decision | Controller | Check against the scan |
|---|---|---|
| Camera model | `OPENCV` (reprojection 0.560 px; `SIMPLE_RADIAL` 1.188 px; `OPENCV_FISHEYE` 0.548 px, only 2% better) | Correct. `OPENCV`: ATE 1.44 cm, point median 1.52 cm. `OPENCV_FISHEYE`: 2.74 cm, 2.69 cm. `SIMPLE_RADIAL`: 30.8 cm, 26.9 cm |
| Floor gate | Warned (coverage 46%) | **False alarm.** The floor-labelled points lie at about -4.92 m against the scan floor at -4.97 m |
| Tapes | Scales disagreed by 14.7%; dropped `height`; kept `horizontal_long` (0.8457 m per unit) | Correct outcome: true scale 0.8432, error +0.3%. The height tape would have been 12.6% low |

### What the second scene shows

- **Model choice held on both scenes.** The comparison rule picked `OPENCV` twice, and `OPENCV`
  was the best model against the scan both times. On this scene an absolute limit of 1 px would
  also have rejected `SIMPLE_RADIAL` (1.19 px); on `office` it would not have (0.98 px), which is
  why the rule compares models.
- **The floor gate does not work as a floor test.** Coverage was 0.47 on `office`, where the
  detected floor was 74 cm above the real one, and 0.46 here, where it is correct. A value that is
  the same for a wrong and a right floor carries no information about the floor. The gate is not
  validated and should not be relied on.
- **The right tape was dropped for the wrong reason.** The height tape was wrong here because the
  reconstructed ceiling plane (181 supporting points) sits above the scan's main ceiling, giving a
  height of 5.25 m against 4.59 m. The floor was fine. The rule "drop height when the floor gate
  warns" produced the correct answer by coincidence.
- **A diffuse wall went undetected.** `wall_2_hi` has a sigma of 0.476 units (about 40 cm), which
  is not a plane, yet it yields the horizontal dimension `horizontal_short` (21.8 units, which
  would imply a scale 54% off). No gate checks plane thickness. A plane-sigma gate is the obvious
  next check, but it is untested.

### Evaluation caveat on this scene

The alignment used as reference (a Sim(3) fitted on camera centres) is tilted by about 0.5 degrees
on the floor and 1 degree on the ceiling here, because the cameras span only 5 x 11 m and 13 cm
vertically while the structure spans 35 to 40 m. That shifts far points by 0.3 to 0.6 m
vertically. Consequently the vertical check and the label-agreement number for this scene are not
valid and are not reported. The camera-model ranking is unaffected (ATE is the residual of the
fit itself), and the floor comparison is accurate to the 5 cm level because it uses medians near
the cameras.

### Status after two scenes

Two scenes are still a small sample, and the controller design was shaped by the first. Only the
camera-model comparison is supported by both. The tape cross-check worked (it caught disagreeing
tapes twice), but its attribution of the culprit is unreliable. The floor gate failed its test.
