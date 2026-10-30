# Day 3: registration experiments (ETH3D `office`)

Goal: align an SfM reconstruction (arbitrary frame and scale) to a reference scan **without
using ground-truth poses**, as a deployable registration stage. Code: `registration/pipeline.py`
(own RANSAC over Sim(3) hypotheses and own trimmed ICP with scale; FPFH features come from Open3D),
comparison harness: `registration/compare.py`. Tests: `tests/test_registration.py` (16 tests in total pass).

Input: the OPENCV_FISHEYE reconstruction from the Day 2 debug (9,484 sparse points, 26 cameras),
reference scan 419,482 points (2 cm voxels). True scale from the camera-centre fit: 0.1534.

## Result 1: the point-distance metric can be gamed

First comparison, scored only by point-to-scan distance:

| Method | Median point distance | Recovered scale (true 0.153) |
|---|---|---|
| Open3D FPFH+RANSAC+ICP with scaling | 0.25 to 0.71 cm | 0.0005 to 0.0017 |
| Own pipeline (no scale bound), seeds 2 and 3 | 0.75 cm | 0.0023 |

These look better than the Day 2 result (1.35 cm) but are degenerate: scaled ICP shrinks the whole
reconstruction onto a small patch of the scan, which minimises a one-sided nearest-neighbour distance.
The scale values are 100x too small. Open3D's own scaled pipeline does this in every successful run
(one more run failed with a NaN transform).

Countermeasures implemented:
- Score every method by where it places the **cameras** relative to the reference poses
  (ground truth is used for evaluation only), next to the point distance.
- `register()` bounds the coarse scale (the clouds are assumed to cover similar extents, so after
  RMS normalisation the scale is near 1) and clamps ICP scale to +-15% of the coarse scale.

## Result 2: with collapse prevented, global registration still fails

| Method | Camera RMSE | Point median | Within 2 cm | Scale |
|---|---|---|---|---|
| Camera-centre Sim(3) (uses GT poses, reference point) | 1.10 cm | 1.35 cm | 82% | 0.1534 |
| Own ICP only, started from that transform | 2.37 cm | 0.83 cm | 95% | 0.1525 |
| Open3D default, 4 usable seeds of 5 | 156 to 341 cm | 0.25 to 0.71 cm | 99 to 100% | 0.0005 to 0.0017 |
| Own full pipeline, 5 seeds | 136 to 557 cm | 1.04 to 23.5 cm | 5 to 70% | 0.142 to 0.234 |

- Runtime of the own pipeline: 13 to 32 s per run; Open3D: 3 to 9 s.
- In three own-pipeline seeds the points sit within about 1 cm of the scan while the cameras are
  1.4 to 5.6 m from their true positions. The scene is a room made of large planes, so many
  placements put points on some plane of the scan. Point distance does not identify the right one.
- The scale is now plausible in most seeds but not reliable (0.142 to 0.234 against 0.153).

## Conclusion

- Registration of sparse SfM points to the scan with FPFH and RANSAC does **not** work on this
  scene, for the library default or for my implementation. I have no working global registration yet.
- Fine registration works when it starts near the answer: ICP from the camera-centre fit improves
  point distance from 1.35 cm to 0.83 cm.
- Camera error after that ICP (2.37 cm) is worse than the camera-centre fit (1.10 cm), so the
  point-optimal transform is not the pose-optimal one. I have not investigated why.

## Hypotheses for why it fails (not yet tested)

1. Sparse SfM points have different density and noise from the scan, so FPFH descriptors rarely match.
2. Planar, repetitive geometry makes feature matches ambiguous.
3. Hypotheses are scored by inliers among putative matches only; scoring on the full cloud
   (or a symmetric measure) may reject wrong placements.

## Next options

- Evaluate on a dense reconstruction instead of the sparse points.
- Use a structural prior for rooms: fit dominant planes, align the Manhattan frame, resolve the
  remaining ambiguity with the scan.
- Resolve scale and placement in deployment from tape-measured distances, which is the real use
  case for the self-captured room.

## Result 3: room-plane (Manhattan) coarse registration

Code: `registration/manhattan.py`, tests: `tests/test_manhattan.py` (22 tests in total pass).
Method: estimate the three dominant orthogonal directions of each cloud from surface normals,
enumerate the 24 axis-aligned rotations relating the two frames, set scale and translation per
candidate (robust bounding-box extents, or per-axis wall-peak histogram correlation), refine with
scale-bounded ICP, and rank candidates by voxel-occupancy IoU (symmetric, so it cannot be
improved by shrinking the source).

Findings on the `office` reconstruction (sparse points vs. the scan):

| Stage | Outcome |
|---|---|
| Axis estimation | Correct. Estimated source axes match the reference-pose frame at |cos| 0.993 to 1.0 |
| Candidate ranking | Works. The candidate with the highest IoU was also the one with the smallest camera error |
| Extent initialisation | Best candidate still 106 cm from the true camera positions; scale 0.187 vs. true 0.153 |
| Histogram initialisation | Worse: 196 cm, and 609 s versus 189 s |
| Runtime (extent mode) | 189 s per run on 420k scan points, deterministic across seeds |

Interpretation: the room-plane idea fixes the **rotation**, which FPFH could not (all five
FPFH seeds were off by metres). It does not fix **translation and scale**. The scale is about 22%
too large, which is what a robust-extent ratio gives when the sparse SfM points cover a smaller
part of the room than the scan does. I have not verified that explanation (for example by cropping
the scan to the cameras' field of view first).

On synthetic rooms the same code recovers scale to 5% and placement to under 10 cm, and the
extent initialisation was already within 0.5% of the truth there. The histogram initialisation
was worse than extent on sparse synthetic input too, and is kept only as an experimental option.

## What this means for the project

- Registration to a laser scan is a benchmark device. For the self-captured demo room there is
  no scan; what is needed is **orientation** (room axes, which this work provides) and **metric
  scale** (from tape-measured distances).
- The remaining registration failure (translation and scale under partial coverage) is therefore
  not on the critical path for the demo, and I am recording it as an open problem instead of tuning
  further against the benchmark.
