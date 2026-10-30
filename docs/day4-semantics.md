# Day 4: structural semantics (ETH3D `office`)

Code: `semantics/structure.py` (room axes, planes, point labels with confidence),
`semantics/scene_graph.py` (planes, objects, relations, `scale_from_reference`),
`semantics/evaluate.py` (evaluation against the scan). Tests: `tests/test_semantics.py`
(27 tests in total pass).

## Method

1. Room axes: the Manhattan frame from Day 3; the vertical axis is the one closest to the
   cameras' mean up vector.
2. Planes: peaks of the point distribution along each axis. **Roles come from the camera
   positions**: the low-end plane is the lowest peak below every camera, the high-end plane the
   highest peak above every camera. A side with no such peak gets no plane and is reported as
   `not_observed` in the scene graph.
3. Labels: each point gets the nearest plane within 3 sigma, otherwise `object`. Confidence
   falls off as a Gaussian of the distance in units of sigma.
4. Scene graph: plane nodes, DBSCAN object clusters with size and height above the floor,
   `perpendicular`, `on` and `against` relations. Metric units come from one tape-measured
   distance (`scale_from_reference`).

## Evaluation protocol

No semantic ground truth exists for ETH3D, so the scan's own planes serve as the reference.
SfM points are moved into the scan frame with a Sim(3) fitted on camera centres against the
reference poses (ground truth, evaluation only). Plane tolerance for judging the points is
3 cm (3 sigma with sigma 1 cm), roughly twice the point error measured on this scene. I picked
that value after seeing that the scan's own sigma (0.6 to 0.9 cm) was tighter than the
reconstruction noise, so it is a judgement call, not a pre-registered threshold.

## Results (9,484 sparse points, one scene)

| Check | Result |
|---|---|
| Label agreement with scan-derived planes | 85% overall |
| Ceiling class (6,928 reference points) | 100% agreement |
| Wall class (561) | 71% |
| Object class (1,985) | 38% |
| Floor class (10 reference points) | 0%, but only 10 points exist |
| Agreement by confidence 0.8 to 1.0 / 0.5 to 0.8 / below 0.5 | 92% / 62% / 33% (7,864 / 1,106 / 514 points) |
| SfM ceiling height in the scan frame | 2.406 vs. scan 2.399 (0.7 cm) |
| SfM "floor" in the scan frame | -0.906 vs. scan floor -1.648 |

Confidence is informative: agreement rises monotonically with it.

### The floor was not the floor

The sparse reconstruction never observed the real floor (textureless carpet; only 10 reference
points lie on it). The "floor" plane it reports is a surface 74 cm above the true floor, most
likely furniture. This shows up when each room dimension is used as the tape measurement:

| Tape dimension | Scale error vs. true scale (0.1534 m per unit) |
|---|---|
| Height (floor to ceiling) | +21.8% |
| Horizontal (wall to wall) | +2.1% |

So with a horizontal tape the metric scale is good to about 2%; with a vertical tape it is not,
and nothing inside the reconstruction flagged the floor as suspect. The pipeline cannot yet
distinguish a real floor from a weakly supported surface that happens to be the lowest one.

## Bugs found on the way

- Plane roles were first assigned by position order only. The weakly observed floor fell below
  the peak threshold, so the ceiling was labelled "floor" and 6,900 ceiling points were
  mislabelled. Using camera positions fixed it.
- `find_peaks` cannot report a peak in the first or last histogram bin, and walls sit exactly
  at the ends of the range. The histogram is now padded with empty bins.
- Plane offsets were histogram bin centres (about 1 cm error) against a 2.5 cm tolerance;
  they are now refined to the median of the supporting points.
- A far outlier in the sparse points inflated the extent and the sigma estimates; extents now
  use the 0.5 and 99.5 percentiles.

## Limits

- One scene, one lens. Agreement numbers on 9.5k points carry no confidence intervals.
- Semantics here are structural only (planes and unlabelled objects). Object categories need a
  2D segmentation model, which has not been run.
- The reference labels come from the scan's planes, not from human annotation; "object" in the
  reference means only "not on a scan plane".
- Walls with one side unobserved (here `wall_2_lo`) are absent from the graph, so the room's
  second horizontal extent is not available from the sparse points.

## Next

- Add a floor-plausibility check (support, height relative to camera positions).
- 2D segmentation lifted to 3D for object categories (needs a model download).
