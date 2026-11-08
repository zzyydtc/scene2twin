# scene2twin

Turn photos of a room into a measured, uncertainty-labelled 3D map, and check the result against
independent reference geometry instead of trusting a pretty visualisation.

The pipeline reconstructs a room with structure-from-motion, finds its floor, ceiling and walls,
lifts 2D instance masks onto the 3D points, sets metric scale from tape measurements, and runs a
set of quality gates that decide which camera model and which measurements to trust. It is
evaluated against laser scans from the ETH3D benchmark.

> Status: research prototype. Several parts work, some did not, and both are documented below.
> A static [web viewer](#viewer) shows the two test scenes with the pipeline's labels, confidences,
> quality gates and decision log.

**Live demo:** <https://zzyydtc.github.io/scene2twin/viewer/> (orbit with the mouse; click a photo or
a camera to see the model drawn over it; Compare splits the view).

## What works and what did not

| Area | Outcome |
|---|---|
| Camera-model choice | A rule that compares candidate models picked `OPENCV` on both test scenes, and `OPENCV` was the most accurate against the laser scan both times (median point error 0.84 cm and 1.52 cm). A single reprojection-error threshold would have accepted a wrong model on one scene (0.98 px) |
| Room structure | Floor, ceiling and wall planes and per-point labels with a confidence that tracks accuracy (92% agreement at confidence 0.8 to 1.0, 33% below 0.5, on one scene) |
| Instance lifting | SAM masks lifted to 3D by cross-view voting; separates keyboards, bags and whiteboards. Produces some room-sized false merges, which a gate catches |
| Registration to a laser scan | FPFH+RANSAC, Open3D's scaled pipeline and a room-axes method all failed to place the reconstruction (Day 3). A later method, FFT correlation of occupancy grids restricted to the covered box plus point-to-plane ICP, placed all three tested reconstructions within 1 to 2 cm of the fit to the reference poses (baseline: 106 cm, 107 cm and 964 cm off); on the large scene it took 93 minutes |
| Floor-plausibility gate | **Failed validation.** It gave the same value for a wrong floor and a correct one |
| Metric scale | Cross-checking two tape measurements caught a bad one on both scenes; the final scale was within 3% and 0.3% of the reference |
| SfM engine | COLMAP's global mapper matched the incremental mapper on `office` (0.73 against 0.74 cm camera error, about 1.8 times faster) but **failed with its defaults on `delivery_area`** (283 cm, two correct halves rotated 167 degrees against each other, with a normal-looking reprojection error); a reference-free filter that drops image pairs whose rotation loops do not close repairs it (2.1 cm, still worse than incremental's 1.44 cm). Letting the principal point be refined helped more than the engine choice: incremental 1.44 to 0.33 cm on `delivery_area`, 0.74 to 0.45 cm on `office`; the controller has it as an opt-in flag (`--refine-principal-point`) because it also flipped a gate decision on `delivery_area` (Day 8). The refined principal point lands within 4 px of the reference calibration's, while the fixed one was 13 to 17 px off |
| Learned matching | DISK keypoints with LightGlue matches (pretrained, through `kornia`) in place of SIFT gave 38% to 57% more points but noisier ones (mean reprojection 1.5 to 2.7 times higher, median point error 11.0 against 8.4 mm on `office`) and lost 5 of 26 and 4 of 44 images, which became islands in the match graph. Camera error was no better on the images both registered (0.57 against 0.59 cm on `office`). Part of the keypoint noise is that DISK reports whole pixels of the shrunken image (Day 9) |
| Feed-forward model | VGGT (pretrained, one pass over all images, no matching) placed the cameras in 7 to 14 s but 12.5% of the scene size off on both scenes (7.0 cm on `office`, 40.4 cm on `delivery_area`; SfM: 0.74 and 1.44 cm). As the starting point for COLMAP's triangulation and bundle adjustment it reached the incremental mapper's result on `delivery_area` (1.47 against 1.44 cm) but not on `office` (3.07 cm), and with the principal point refined produced a wrong model with a 0.38 px reprojection error there (Day 10) |
| MASt3R-SfM | Pairwise network predictions aligned into one scene were closer than VGGT on `office` (4.55 against 7.0 cm) but failed on `delivery_area`: four images were placed about 97 degrees off the rest (217 cm overall, 0.54 px reprojection error after COLMAP refinement of a still-wrong model). The same four images defeat the learned matcher of Day 9, and SIFT links them to the rest (Day 11) |
| Predicted precision | The precision a triangulation predicts for its own points is too optimistic (1.6 to 2.7 times too small, heavy tails), and COLMAP's reprojection error ranks real error barely better than a random order on one scene (Day 7) |

Details, numbers and the bugs found along the way are in [`docs/`](docs/):
[camera-model debug](docs/debug-day2-camera-model.md),
[registration](docs/day3-registration.md),
[semantics](docs/day4-semantics.md),
[SAM lifting](docs/day4b-sam-lift.md),
[controller and second scene](docs/day5-agent.md),
[registration by correlation and predicted-vs-observed precision](docs/day7-comparisons.md),
[incremental against global SfM](docs/day8-sfm-engines.md),
[learned matching against SIFT](docs/day9-learned-matching.md),
[VGGT, a feed-forward model, against SfM](docs/day10-vggt.md),
[MASt3R-SfM against SfM and VGGT](docs/day11-mast3r.md).

Limits that apply to every number above: two scenes, one camera each, thresholds chosen after
seeing the first scene, and no human-annotated semantics (the scan's own planes are the reference).

## Authorship: what is written here and what is a library

| Component | Source |
|---|---|
| SfM (feature matching, incremental mapping, bundle adjustment) | COLMAP through `pycolmap` |
| Pairwise 3D prediction and alignment (MASt3R-SfM, pretrained) | code cloned to `data/third_party/mast3r`, weights downloaded separately (CC BY-NC-SA 4.0, checkpoint trained on non-commercial datasets); the runner and scoring are written here (`feedforward/mast3r_run.py`) |
| Feed-forward cameras and depth (VGGT-1B) | pretrained, code cloned to `data/third_party/vggt` and weights downloaded separately (Meta's VGGT licence); the runner, scoring and the COLMAP hand-over are written here (`feedforward/`) |
| Learned keypoints and matcher (DISK, LightGlue) | pretrained, through `kornia`; the code that feeds them into a COLMAP database is written here (`matching/`) |
| 2D instance masks | SAM ViT-B, pretrained, not trained here |
| FPFH features, normal estimation, voxel filtering, DBSCAN, the library registration baseline | Open3D |
| Sim(3) alignment (Umeyama), RANSAC over Sim(3), scale-bounded trimmed ICP, Manhattan-frame registration, occupancy-IoU ranking, FFT occupancy-grid correlation, point-to-plane ICP with scale | written here (`registration/`) |
| Pose, point-cloud and reprojection metrics; scan loading and alignment; triangulation-covariance calibration against the scan | written here (`eval/`) |
| Room axes, plane detection with camera-based roles, point labelling and confidence, scene graph, cross-view instance voting | written here (`semantics/`) |
| Quality gates, camera-model comparison, tape cross-check, decision-logging controller | written here (`agent/`) |

The controller is a deterministic rule-based program; no language model is involved.

## Layout

```
eval/          metrics against ground truth, scan loading, precision calibration, SfM engine comparison
registration/  Sim(3), RANSAC, ICP, Manhattan frame, raster correlation, comparison harness
sfm/           view-graph filter for global SfM
matching/      learned matching (DISK + LightGlue) into a COLMAP database
feedforward/   VGGT and MASt3R-SfM runners, learned-camera-initialised COLMAP refinement
semantics/     structure, scene graph, SAM lifting, evaluation against the scan
agent/         quality gates and the controller
scripts/       baseline reconstruction, viewer export
tests/         unit tests (82)
docs/          debug logs and findings
viewer/        static Three.js viewer and its exported data (viewer/data)
data/          local only, git-ignored (datasets, model weights)
outputs/       local only, git-ignored
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pytest
```

PyTorch is only needed for the SAM step; pick the build that matches your CUDA version.

## Reproducing the results

Data is not included. ETH3D scenes come from <https://www.eth3d.net/datasets> (files named
`<scene>_dslr_jpg.7z` and `<scene>_scan_clean.7z`, extract with `py7zr` or 7-Zip into
`data/eth3d/<scene>/`). The SAM ViT-B checkpoint is
`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` (375 MB), saved to
`data/models/`.

```
# controller: reconstruct with three camera models, choose one, derive scale from tapes
python -m agent.controller --images data/eth3d/office/images/dslr_images \
    --out outputs/office_agent --tape height=4.05 --tape horizontal=3.13

# accuracy of a reconstruction against the laser scan
python -m eval.run_eval --model outputs/office_agent/opencv/best \
    --scene data/eth3d/office --out outputs/office_agent/opencv/eval.json

# structural semantics against the scan, and SAM lifting
python -m semantics.evaluate --model outputs/office_agent/opencv/best \
    --scene data/eth3d/office --out outputs/office_agent/opencv/semantics.json
python -m semantics.run_sam --model outputs/office_agent/opencv/best \
    --images data/eth3d/office/images/dslr_images \
    --checkpoint data/models/sam_vit_b_01ec64.pth --out outputs/office_agent/sam

# registration to the scan, baseline against the raster method (add --reg-voxel 0.1 for a large scene)
python -m registration.compare --model outputs/office_agent/opencv/best --scene data/eth3d/office \
    --out outputs/compare/office.json --seeds 1 --methods ours_manhattan,ours_raster

# incremental against global SfM on the controller's matches
python -m eval.compare_sfm --database outputs/office_agent/opencv/database.db \
    --images data/eth3d/office/images/dslr_images --scene data/eth3d/office --out outputs/sfm_engines/office
# engines: incremental, global, global_filtered; add --refine-principal-point to let all refine it

# learned matching: build a database with DISK + LightGlue, then map it with the commands above
python -m matching.learned --images data/eth3d/office/images/dslr_images \n    --database outputs/learned/office/database.db --max-side 3200 --max-keypoints 8192

# VGGT: clone https://github.com/facebookresearch/vggt to data/third_party/vggt, download model.safetensors of
# facebook/VGGT-1B to data/models/vggt-1b/, then run it alone and as the start for COLMAP
python -m feedforward.vggt_run --images data/eth3d/office/images/dslr_images \n    --scene data/eth3d/office --out outputs/vggt/office
python -m feedforward.refine --vggt outputs/vggt/office --database outputs/office_agent/opencv/database.db \n    --images data/eth3d/office/images/dslr_images --scene data/eth3d/office --out outputs/vggt/office_refined

# MASt3R-SfM: clone https://github.com/naver/mast3r (with --recursive) to data/third_party/mast3r and download
# naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric to data/models/mast3r-vitl-512/ (about 80 minutes on delivery_area)
python -m feedforward.mast3r_run --images data/eth3d/office/images/dslr_images \n    --scene data/eth3d/office --out outputs/mast3r/office

# predicted precision against observed error
python -m eval.run_calibration --model outputs/office_agent/opencv/best --scene data/eth3d/office \
    --out outputs/calibration/office.json --figure docs/figures/office_calibration.png
```

The tape values above were taken from the laser scan to simulate a measured room. For a real room
they are distances measured with a tape between the walls or floor and ceiling.

## Viewer

`viewer/index.html` is a static page (Three.js from a CDN, no build step). It shows the
reconstruction, and optionally the laser scan once the reconstruction has been registered to it.

- **Colouring:** photo colour, structure label, label confidence, SAM instance, the precision each
  point's triangulation geometry predicts, or its observed distance to the laser scan (Day 7).
- **Laser scan (Day 7):** a layer showing the scan around the reconstruction, with a switch for
  which registration places it: the new occupancy-grid method or the Day 3 room-axes baseline, with
  the camera error each leaves. The scan is thinned to at most 100,000 points and loaded on demand.
- **Layers (checkboxes):** the points, each structure class on its own, the fitted room planes,
  camera positions, and the photos placed in 3D at their cameras.
- **Compare:** splits the view with a draggable divider and colours the left and right sides
  differently, from the same viewpoint (for example structure against photo colour).
- **Original photos:** pick a photo, or click a camera in the 3D view. The photo opens with the
  reconstructed points drawn over it (the keypoints that were triangulated into 3D), coloured by
  structure, confidence or instance. "View from here" moves the 3D camera to the photo's pose, so
  you can compare the photo with the model from the same viewpoint; this ignores lens distortion
  and levels the roll, so it is approximate.
- **Inspect:** click a point to read what the pipeline inferred for it. The panel also lists the
  quality gates, the camera-model comparison, the scale basis and the controller's decision log.

```
# export a controller run (writes viewer/data/<name>/, with downscaled photos)
python -m scripts.export_viewer --report outputs/office_agent/report.json --name office \
    --instance-labels outputs/office_agent/sam/instance_labels.npy \
    --images data/eth3d/office/images/dslr_images
python -m scripts.export_viewer --report outputs/delivery_agent/report.json --name delivery_area \
    --images data/eth3d/delivery_area/images/dslr_images

# Day 7 data (predicted and observed precision, the scan placed by two registrations): add
#   --scene data/eth3d/office --placement raster:outputs/placement/office.json:ours_raster_prior \
#   --placement baseline:outputs/placement/office.json:ours_manhattan
# to the commands above. registration.compare writes the transforms into its JSON.

# serve the repo root and open http://localhost:8000/viewer/
python -m http.server 8000
```

`viewer/data` holds a sparse reconstruction and 800 px copies of the ETH3D photos and, for the Day 7
comparisons, per-point precision and error plus a thinned copy of the laser scan around each
reconstruction (about 8 MB for both scenes). It is derived from ETH3D data and is licensed under CC BY-NC-SA 4.0, not the MIT
licence of the code; see [viewer/data/LICENSE](viewer/data/LICENSE).

## License

- **Code:** MIT. See [LICENSE](LICENSE).
- **`viewer/data/`** (downscaled ETH3D photos and the points, labels and keypoints derived from
  them): [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), inherited from
  ETH3D. Non-commercial use only, with attribution, and adaptations must be shared under the same
  licence. The labels, confidences and instance ids in it were computed by this project's code.
- **SAM:** the Segment Anything code and model weights are Apache 2.0 (Meta AI). The weights are
  downloaded separately and are not in this repository.

## Credits and citation

The test scenes (`office`, `delivery_area`) and the reference laser scans are from the ETH3D
benchmark (<https://www.eth3d.net>). If you use this work, cite the ETH3D publication for the
high-resolution multi-view data:

> T. Schöps, J. L. Schönberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys and A. Geiger.
> A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos. CVPR 2017.

Other work used: COLMAP (J. L. Schönberger and J.-M. Frahm, *Structure-from-Motion Revisited*,
CVPR 2016) through `pycolmap`; Segment Anything (A. Kirillov et al., ICCV 2023); Open3D (Q.-Y. Zhou,
J. Park and V. Koltun, 2018); Three.js for the viewer.
