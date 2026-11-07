# Day 9: learned feature matching (DISK + LightGlue) against SIFT

Question: does replacing COLMAP's SIFT features and ratio-test matching with a learned detector and
matcher give a better reconstruction? Everything after matching is unchanged: COLMAP verifies the
pairs geometrically, then the same incremental and global mappers run and the result is scored against
the ETH3D reference poses and laser scan with the same metrics as before.

## What was run

- `matching/learned.py` writes a COLMAP database: DISK keypoints (up to 8,192 per image) and LightGlue
  matches for every image pair, then COLMAP's own geometric verification. Both networks are pretrained
  and come from `kornia`; nothing is trained here. The cameras are the same single OPENCV camera the
  controller uses.
- Images are shrunk to a longer side of 3,200 px before DISK, the size COLMAP's SIFT works at by default
  (the photos are 6,048 x 4,032). Building the database took about 12 minutes for `office` (26 images)
  and 20 minutes for `delivery_area` (44 images) on the laptop GPU, single runs; I did not time the SIFT
  step separately, so this is not a speed comparison.
- `eval.compare_sfm` then maps that database. The incremental mapper returned the same model for every
  seed in Day 8, so one run per setting is reported here.

## Result

Camera RMSE is a fit of the camera centres to the reference poses; "common" restricts both models to the
images that both registered, because a model with fewer, easier images is not comparable.

| | SIFT, principal point fixed | DISK + LightGlue, fixed | SIFT, principal point refined | DISK + LightGlue, refined |
|---|---|---|---|---|
| **`office`** images registered | 26 | 21 | 26 | 21 |
| camera RMSE, all (cm) | 0.74 | 0.57 | 0.45 (3-seed mean, Day 8) | 0.37 |
| camera RMSE, 21 common (cm) | 0.59 | 0.57 | not computed | not computed |
| 3D points | 9,479 | 13,094 | | |
| median point error to the scan (mm) | 8.4 | 11.0 | | |
| points within 2 cm of the scan | 95.3% | 72.0% | | |
| mean reprojection error (px) | 0.42 | 1.14 | | |
| **`delivery_area`** images registered | 44 | 40 | 44 | 40 |
| camera RMSE, all (cm) | 1.44 | 1.05 | 0.34 | 0.48 |
| camera RMSE, 40 common (cm) | 1.29 | 1.05 | 0.29 | 0.48 |
| 3D points | 28,961 | 45,412 | 29,074 | 45,403 |
| median point error to the scan (mm) | 15.2 | 18.1 | 10.7 | 12.1 |
| points within 2 cm of the scan | 58.0% | 52.7% | 70.3% | 63.4% |
| mean reprojection error (px) | 0.56 | 0.85 | 0.53 | 0.84 |

- **Learned matching is not better overall.** It gives 38% (`office`) and 57% (`delivery_area`) more 3D
  points, but they sit further from the scan, the mean reprojection error is 1.5 to 2.7 times higher, and
  it loses images (5 of 26 and 4 of 44). Where the camera error is lower with a fixed principal point
  (`delivery_area`, 1.05 against 1.29 cm on the common images), it is higher once the principal point is
  refined (0.48 against 0.29 cm).
- On `office` the camera error on the common images is the same (0.57 and 0.59 cm). The 0.57 against
  0.74 cm in the "all" row is the missing images, not a better reconstruction.
- With the filtered global mapper the learned matches give 1.54 cm on `delivery_area` (incremental 1.05)
  and 16.2 cm on `office` (see below).

Image size matters. Incremental mapping on `office`, same database recipe, one run each:

| DISK input size, keypoints | Images registered | Camera RMSE, all (cm) | Camera RMSE, 21 common (cm) |
|---|---|---|---|
| 1,600 px, 4,096 | 22 | 0.83 | not computed |
| 1,600 px, 8,192 | 23 | 2.07 | 0.55 |
| 2,400 px, 8,192 | 23 | 3.53 | not computed |
| 3,200 px, 8,192 | 21 | 0.57 | 0.57 |

The camera errors across rows are over different image sets. The 2.07 and 3.53 cm come from the two
extra images that the 3,200 px run left out, which are placed badly.

## Where the error comes from

**1. The keypoints are less precise than SIFT's.** For every inlier match of a pair that agrees with the
reference (at least half the matches within 3 px of the epipolar line the reference poses and the
reference lens calibration predict), the distance to that line:

| | median | 90th percentile |
|---|---|---|
| `office`, SIFT | 0.21 px | 0.77 px |
| `office`, DISK 3,200 px | 0.77 px | 2.10 px |
| `office`, DISK 1,600 px | 1.14 px | 2.47 px |
| `delivery_area`, SIFT | 0.23 px | 0.92 px |
| `delivery_area`, DISK 3,200 px | 0.50 px | 1.30 px |

This is in full-size (6,048 px wide) pixels, and it is roughly the localisation error of the two
keypoints of a match. The reprojection error after mapping follows it (table above).

**2. Part of it is that DISK reports integer pixels of the shrunken image.** The keypoints of the 3,200
px run lie on whole pixels of the resized image (largest deviation 0.0002 px), which is a grid of 1.9
px at full size. Rounding SIFT's keypoints to the same grid, with nothing else changed, raises its
`office` error from 0.21 px to 0.48 px (3,200 px grid), 0.60 px (2,400 px) and 0.84 px (1,600 px).
That is about half of DISK's extra error at 3,200 px (0.27 of 0.55 px) and most of it at 1,600 px
(0.63 of 0.93 px). The rest is not explained by this check; it is DISK's own localisation or its
matches. A sub-pixel refinement with `cv2.cornerSubPix` on the full-size image made it worse (0.77
to 0.85 and 0.88 px), so it is not in the code. Running DISK at the full 6,048 px was not tried: at 3,200
px the process already held nearly all of the 16 GB of GPU memory.

**3. The lost images are islands in the match graph.**

- `delivery_area`: images DSC_0711 to DSC_0714 have **no** verified pair with the other 40 images in the
  DISK database, so no mapper can place them. With SIFT each of the four has 16 to 20 verified pairs.
- `office`: images DSC_0237 to DSC_0241 are linked to the rest by two pairs, with 15 and 49 inliers; with
  SIFT each has 5 to 11 pairs. Incremental mapping leaves all five out.

On `delivery_area` the four images match each other very well (8,700 to 11,400 inliers per image) but not the rest of the scene. I did
not find out what these images show or why LightGlue fails to bridge them; it is a loss of wide-baseline
or large-viewpoint-change matches that SIFT with exhaustive matching keeps.

**4. Fewer matched pairs, more inliers per pair, not fewer wrong pairs.** Pairs with verified inliers:
`office` 136 against 170 for SIFT, `delivery_area` 311 against 385. The inliers in agreeing pairs are
1.9 and 2.2 times as many. Pairs that disagree with the reference: `office` 1 of 136 (SIFT 0 of 170),
`delivery_area` 36 of 311 (SIFT 69 of 385), so learned matching has a lower rate there (12% against
18%) but not a clean one.

**5. The global mapper does worse on the learned matches, and mostly because of one image.** On `office`
the global mapper gives 18.5 cm and the filtered one 16.2 cm (23 images). In the filtered model one
camera, DSC_0241 (a single 15-inlier link), is 73 cm off and the next worst is 10 cm; without the five
worst cameras the RMS is 4.4 cm, still much more than the incremental mapper's 0.57 cm. At 1,600 px the
filtered global mapper gave 1.29 cm. With the principal point refined the filtered global mapper on 3,200
px matches gave 56 cm. These are single runs of a mapper that varies by seed, so the size of these
numbers is not reliable; what they show is that a weakly attached image can ruin a global solution
while the incremental mapper simply leaves it out.

## An error in the Day 8 diagnosis, found here

The pair analysis above uses the **reference lens calibration**. The Day 8 analysis used the camera
COLMAP estimated, with its principal point fixed at the image centre. The reference principal point
is 13 to 17 px away from the centre, and with that wrong camera the reference said many good pairs were
wrong: on `office`, 67 of the 170 SIFT pairs (with the reference camera, 0). On `delivery_area` the
corrected count is 69 wrong pairs, not 110, and all 69 have fewer than 50 inliers. Day 8's
`docs/day8-sfm-engines.md` has been corrected accordingly. The same check also answered an open question
of Day 8: the refined principal point lands within 4 px (`office`) and 1 px (`delivery_area`) of the
reference one, so refining it removes a real 13 to 17 px error and is not just absorbing something else.

## What this shows

- **For these two scenes SIFT with exhaustive matching is the better front end** for a COLMAP
  reconstruction: it registers every image, has sub-pixel keypoints, and gives the lowest error once the
  principal point is right. DISK + LightGlue produces more points, but they are noisier, and it breaks
  the match graph.
- **The keypoint grid is a fixable cause, not the whole one.** A matcher that returned sub-pixel
  keypoints, or running DISK at full size, could close part of the gap; this was not tested.
- **A reference check is only as good as its calibration.** The Day 8 labels were wrong because of a
  13 px lens parameter; the numbers that drove the conclusions changed, the conclusions held.

## Limits

- Two scenes, one camera each, one matcher (DISK + LightGlue as packaged in `kornia` 0.9, default
  settings, 8,192 keypoints), one run per setting for the learned matches. No tuning of LightGlue's
  thresholds or of the number of keypoints beyond what is in the tables.
- The learned matcher is not wired into the controller, which still uses SIFT; nothing downstream was
  rerun on these models (gates, structure, scale).
- Image names (which images are the islands) are reported, but I did not look at them. The explanation
  of the lost images as a broken match graph is measured; why it breaks is not.
- The pair and keypoint analysis scripts are local scratch (`outputs/`, not in the repository). The
  numbers above come from them; `matching/learned.py` and `eval/compare_sfm.py` are in the repository.
