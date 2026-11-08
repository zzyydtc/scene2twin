# Day 11: MASt3R-SfM against structure-from-motion and VGGT

Question: MASt3R (Leroy et al., 2024) is the other kind of learned reconstruction model: a network
that, for a *pair* of images, predicts a 3D point for every pixel and a descriptor for matching. Its
SfM mode runs it over image pairs and aligns the pairs into one scene with an optimisation, so it uses
matches (of its own) and a global optimisation, unlike VGGT's single pass. Does it reconstruct the
two scenes better than VGGT or than SfM?

## What was run

- `feedforward/mast3r_run.py`: `sparse_global_alignment` from the MASt3R code, with the pretrained
  `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` checkpoint, the complete pair graph (every pair,
  both directions: 650 passes for `office`, 1,892 for `delivery_area`), images at 512 px on the long
  side (512 x 336), one shared focal length (the scenes have one camera), and the demo's default
  optimisation settings (500 coarse and 200 refinement iterations). Cameras and points are scored with the
  same Sim(3) fit and metrics as every other model here.
- `feedforward/refine.py` (Day 10), unchanged, gives MASt3R's cameras and focal length to COLMAP as
  the starting point for triangulation and bundle adjustment.
- The MASt3R code (with its dust3r submodule) is cloned to `data/third_party/mast3r` and the 2.7 GB
  weights are downloaded to `data/models/mast3r-vitl-512`; neither is in the repository. The code is
  CC BY-NC-SA 4.0, and the checkpoint notice says the weights were trained on datasets with
  non-commercial terms, so this is a research comparison only.
- One scipy incompatibility had to be worked around (`scipy.cluster.hierarchy.distance`, which newer
  scipy no longer exposes); this is done in the runner, not in the cloned code.

## Result

Camera RMSE in cm against the reference poses, with the number of images registered when it is not all.

| Method | `office` | `delivery_area` |
|---|---|---|
| SfM, SIFT, incremental (Day 8) | 0.74 | 1.44 |
| SfM, SIFT, global mapper with the loop filter (Day 8) | 0.72 | 2.11 |
| SfM, DISK + LightGlue, incremental (Day 9) | 0.57 (21 of 26) | 1.05 (40 of 44) |
| VGGT alone (Day 10) | 7.0 | 40.4 |
| VGGT cameras as the start for COLMAP (Day 10) | 3.07 | 1.47 |
| **MASt3R-SfM alone** | **4.55** | **217 (failed)** |
| **MASt3R cameras as the start for COLMAP** | **11.39** | **197.6 (failed)** |

### `office`

| | MASt3R-SfM | VGGT |
|---|---|---|
| Camera RMSE | 4.55 cm (8.2% of the 55.7 cm camera spread) | 7.0 cm (12.6%) |
| Median camera error | 2.73 cm | 3.54 cm |
| Median / largest rotation error | 1.37 / 3.85 deg | 1.33 / 3.45 deg |
| Focal length against the reference 3437.8 | 3501.8 (+1.9%) | 3456.6 (+0.5%) |
| Time | about 9 min for the 650 network passes and about 10 min for the alignment | 6.8 s |
| Peak GPU memory | 4.2 GB (alignment) | 9.8 GB |
| Sparse points (1.43 million), median error to the scan, within 2 cm | 79 mm, 9.7% | not produced |
| Dense points, most confident 50%: median error, within 2 cm | 92 mm, 7.2% | 73 mm, 20.3% |
| Dense points, most confident 10% | 44 mm, 17.5% | 18.5 mm, 54.5% |

- **MASt3R-SfM is more accurate than VGGT on the cameras here** (4.55 against 7.0 cm) and costs over 100
  times as much time. It is still about 6 times worse than SfM (0.74 cm), and its points are worse
  than VGGT's confident ones.
- **Its largest camera errors are the five-image group DSC_0237 to 0241** that every method has
  struggled with (Days 9 and 10): DSC_0238, 0239, 0240 and 0237 are 7 to 9 cm off, as the worst four.
- **The "metric" checkpoint is not metric enough to use.** The Sim(3) fit needed a scale of 0.87, so
  the scene came out 15% too large. The tape-based scale in the controller is within 3% and 0.3%.
- **As the start for COLMAP it is worse than raw** (11.39 cm, and 13.9 cm with the principal point
  refined): the refinement started from MASt3R's cameras ended with only 2,623 triangulated points
  and the worst cameras off by 17 to 41 cm (DSC_0230, 0238, 0237, 0241), where VGGT's start ended at
  3.07 cm.

### `delivery_area`: MASt3R-SfM fails

- **Camera RMSE 217 cm and a median rotation error of 112 degrees** over all 44 images, against SfM's
  1.44 cm. The cause is four images: DSC_0711 to DSC_0714, the group that SIFT links to the rest by 16 to
  20 verified pairs per image and the DISK + LightGlue matches did not link at all (Day 9). The four
  agree with each other (0.29 cm RMS among themselves) but sit up to 97 degrees rotated against the
  other 40, and at a different scale (2.79 against 2.19).
- **The other 40 cameras are 41.7 cm RMS with a median rotation error of 2.4 degrees.** This subset
  was chosen afterwards, using the reference, to separate the failure from the rest; it is not a score
  a user could have obtained. It is about as good as VGGT's 40.4 cm over 44 images, and about 30 times worse than SfM.
- **Starting COLMAP from this does not repair it** (197.6 cm, median rotation error 72 degrees; 198.1
  cm with the principal point refined). The bundle adjustment settled in a consistent but wrong
  model: mean reprojection error 0.54 and 0.52 px, 30,000 points, a camera error two metres.
- The focal length is again close (3467.4 against 3408.6, +1.7%), and the fit scale of the 40 good
  cameras (2.19) differs from `office`'s 0.87, so the metric output is not consistent between scenes.
- The 1,892 network passes took about 50 minutes and the alignment about 30. Points were not scored on
  this scene. The model is wrong as a whole, and the scoring did not finish: the first run spent over an
  hour in the nearest-neighbour search against the scan with the sparse points, and a second run
  that subsampled them to 300,000 had not finished after more than ten minutes either. I stopped both and
  did not find the cause. The runner now also drops non-finite points and points far outside the scan's
  box before scoring (with a unit test), but I did not rerun the 80-minute reconstruction to see
  whether that fixes it.

## What this shows

- **MASt3R-SfM is the best of the two learned models on `office` and the worst on `delivery_area`.**
  VGGT placed all 44 images of `delivery_area` and its cameras were 40 cm off; MASt3R-SfM placed 40 of
  them about as well and a four-image group wrongly, with no sign of it in its own loss or in a
  reprojection error of 0.54 px after the refinement.
- **The same four images, DSC_0711 to 0714, defeat three different learned approaches** (the matcher
  of Day 9 and this one fail on them; VGGT's COLMAP-refined result has them as its four worst cameras, by 2.3 cm against a 1.5 cm median), and
  SIFT with exhaustive matching does not. Likewise the group DSC_0237 to 0241 on `office`. A group of
  frames that look at the scene from a different side is where the learned front ends are weakest here.
- **For these scenes the order stays SfM with SIFT first, then VGGT as a coarse start, then MASt3R-SfM.**
  Neither learned model beats SfM on accuracy, robustness or cost on this data, and SfM's output
  contains the information (verified matches, reprojection error) that a gate can check, while both
  learned models' failures were invisible in their own reported error.
- Where I would not generalise: both scenes are ones where classical SfM works. These results say
  nothing about low-overlap, textureless or very few image captures, which is where learned models are
  claimed to help.

## Limits

- Two scenes, one camera each, one checkpoint, the complete graph and the demo's optimisation settings
  only, one run each. The alignment is stochastic in principle; I did not repeat it. Different pair
  graphs (retrieval, windowed), image sizes or more iterations were not tried and might change
  the `delivery_area` result.
- Times are wall clock on a laptop GPU (RTX 3080 Laptop) from single runs. `office` timing is from two
  runs: the network passes from the first (which then stopped on the scipy error) and the alignment
  from the second, which reused the cached passes.
- The `office` sparse points are scored over all 1.43 million; the other point numbers use 300,000.
- The 40-camera subset on `delivery_area` and the comparison of the four images were defined with
  the reference after seeing the failure.
- The per-camera and rotation-group analysis comes from scratch scripts in `outputs/`, not from the
  repository.
