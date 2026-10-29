# Findings log

## Day 2: point accuracy offset traced to the camera model (ETH3D office)

**Symptom.** Baseline SfM (COLMAP default `SIMPLE_RADIAL`) gave 0.99 px mean reprojection
error and 2.3 cm camera-position error, but the sparse points sat a median 8.1 cm from the
laser scan (9% within 2 cm).

**Checks.**
1. Reference `points3D.txt` vs. the scan: median 1.0 cm, 75% within 2 cm. The scan
   alignment is correct, so the offset comes from the reconstruction.
2. Baseline points refined with scaled point-to-point ICP: median 1.0 cm, but ICP had to
   change the scale by 2.4%. The camera-centre Sim(3) was therefore biased in scale.
3. Re-run with `OPENCV_FISHEYE` (the real lens is a fisheye model, `THIN_PRISM_FISHEYE`):

| Camera model | ATE RMSE | Point median to scan | Within 2 cm | Reproj. mean |
|---|---|---|---|---|
| SIMPLE_RADIAL | 2.3 cm | 8.1 cm | 9% | 0.99 px |
| OPENCV_FISHEYE | 1.1 cm | 1.35 cm | ~70% | 0.45 px |

**Cause.** Lens-model mismatch distorted the reconstructed structure while keeping the
reprojection error low, so reprojection error alone did not reveal the problem.

**Side effect found on the way.** Incremental mapping split the fisheye run into two
models (7 and 26 images). The baseline script now keeps the largest model.

**Implication for the agent.** A quality gate must compare against independent
reference geometry, not only reprojection error, and must consider camera-model
alternatives when the two disagree.
