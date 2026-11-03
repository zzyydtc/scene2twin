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

## What works and what did not

| Area | Outcome |
|---|---|
| Camera-model choice | A rule that compares candidate models picked `OPENCV` on both test scenes, and `OPENCV` was the most accurate against the laser scan both times (median point error 0.84 cm and 1.52 cm). A single reprojection-error threshold would have accepted a wrong model on one scene (0.98 px) |
| Room structure | Floor, ceiling and wall planes and per-point labels with a confidence that tracks accuracy (92% agreement at confidence 0.8 to 1.0, 33% below 0.5, on one scene) |
| Instance lifting | SAM masks lifted to 3D by cross-view voting; separates keyboards, bags and whiteboards. Produces some room-sized false merges, which a gate catches |
| Registration to a laser scan | **Did not work.** FPFH+RANSAC, Open3D's scaled pipeline and a room-axes method all failed to place the reconstruction. The room-axes method recovers orientation, not translation or scale |
| Floor-plausibility gate | **Failed validation.** It gave the same value for a wrong floor and a correct one |
| Metric scale | Cross-checking two tape measurements caught a bad one on both scenes; the final scale was within 3% and 0.3% of the reference |

Details, numbers and the bugs found along the way are in [`docs/`](docs/):
[camera-model debug](docs/debug-day2-camera-model.md),
[registration](docs/day3-registration.md),
[semantics](docs/day4-semantics.md),
[SAM lifting](docs/day4b-sam-lift.md),
[controller and second scene](docs/day5-agent.md).

Limits that apply to every number above: two scenes, one camera each, thresholds chosen after
seeing the first scene, and no human-annotated semantics (the scan's own planes are the reference).

## Authorship: what is written here and what is a library

| Component | Source |
|---|---|
| SfM (feature matching, incremental mapping, bundle adjustment) | COLMAP through `pycolmap` |
| 2D instance masks | SAM ViT-B, pretrained, not trained here |
| FPFH features, normal estimation, voxel filtering, DBSCAN, the library registration baseline | Open3D |
| Sim(3) alignment (Umeyama), RANSAC over Sim(3), scale-bounded trimmed ICP, Manhattan-frame registration, occupancy-IoU ranking | written here (`registration/`) |
| Pose, point-cloud and reprojection metrics; scan loading and alignment | written here (`eval/`) |
| Room axes, plane detection with camera-based roles, point labelling and confidence, scene graph, cross-view instance voting | written here (`semantics/`) |
| Quality gates, camera-model comparison, tape cross-check, decision-logging controller | written here (`agent/`) |

The controller is a deterministic rule-based program; no language model is involved.

## Layout

```
eval/          metrics against ground truth, scan loading
registration/  Sim(3), RANSAC, ICP, Manhattan frame, comparison harness
semantics/     structure, scene graph, SAM lifting, evaluation against the scan
agent/         quality gates and the controller
scripts/       baseline reconstruction, viewer export
tests/         unit tests (46)
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
```

The tape values above were taken from the laser scan to simulate a measured room. For a real room
they are distances measured with a tape between the walls or floor and ceiling.

## Viewer

`viewer/index.html` is a static page (Three.js from a CDN, no build step). It shows the
reconstruction only; the laser scan is not shown because registration to it did not work.

- **Colouring:** photo colour, structure label, label confidence or SAM instance.
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

# serve the repo root and open http://localhost:8000/viewer/
python -m http.server 8000
```

`viewer/data` holds a sparse reconstruction and 800 px copies of the ETH3D photos (about 5 MB for
both scenes). It is derived from ETH3D data and is licensed under CC BY-NC-SA 4.0, not the MIT
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
