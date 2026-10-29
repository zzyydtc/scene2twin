# scene2twin

Phone video in, measured and uncertainty-labeled 3D semantic map out.

An agent orchestrates reconstruction, registration, semantic lifting and evaluation, checks its own results against reference measurements, and retries or switches methods when quality gates fail.

> Status: work in progress (private).

## What this project is

- A pipeline: video/images -> camera poses and dense geometry -> alignment to a metric frame -> semantic labels lifted to 3D -> scene graph.
- An evaluation harness: every stage reports error against ground truth (laser scans, tape measurements), not just a visualization.
- An agent layer: runs the stages as tools, inspects quality metrics, decides to retry or switch method, and writes a report separating **measured** structure from **predicted** structure.

## Authorship (what is mine vs. what is a library)

| Component | Source |
|---|---|
| Structure-from-Motion / MVS | COLMAP (library) |
| 2D segmentation | SAM (pretrained model) |
| Evaluation metrics (pose error, accuracy/completeness, reprojection stats) | written in this repo (`eval/`) |
| Sim(3) alignment, coarse-to-fine registration, comparison against library defaults | written in this repo (`registration/`) |
| 2D-to-3D semantic lifting with per-point confidence | written in this repo (`semantics/`) |
| Scene graph and dimension extraction | written in this repo (`semantics/`) |
| Agent orchestration, quality gates, report generation | written in this repo (`agent/`) |
| Web viewer | written in this repo (`viewer/`) |

## Layout

```
eval/          metrics against ground truth
registration/  Sim(3), FPFH+RANSAC, ICP, comparisons
semantics/     2D->3D lifting, uncertainty, scene graph
agent/         tool orchestration, quality gates, reports
viewer/        static web viewer (point cloud, error and uncertainty views)
scripts/       setup and per-day run scripts
data/          local only, git-ignored
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

COLMAP must be installed separately and available on `PATH`.

## Evaluation data

- ETH3D multi-view benchmark scenes (laser-scan ground truth) for quantitative results.
- A self-captured room with tape-measured dimensions for the demo.

## License

TBD before making the repository public.
