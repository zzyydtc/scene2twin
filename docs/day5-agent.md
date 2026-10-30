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
