"""Self-checking controller for the scene2twin pipeline.

Runs reconstruction with several camera models, applies scan-free quality gates,
chooses a model, derives a metric scale from tape measurements, and writes a report
that separates measured structure from inferences. It is a deterministic controller:
every decision is a rule over gate results and is recorded in the decision log. No
language model is involved.

Usage (from the repo root):
    python -m agent.controller --images data/eth3d/office/images/dslr_images \
        --out outputs/office_agent --tape height=4.05 --tape horizontal_1=3.13
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from agent.gates import (
    MODEL_ORDER, Gate, canonical_dimensions, choose_model, floor_coverage, gate_floor, gate_instance_extent,
    gate_registration, gate_tape_consistency,
)
from agent.tools import reconstruct
from eval.colmap_io import read_model_points, read_model_poses
from semantics.evaluate import analyse, dimensions
from semantics.scene_graph import build_scene_graph


class Log:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(self, step: str, observation: str, decision: str) -> None:
        self.entries.append({"step": step, "observation": observation, "decision": decision})
        print(f"[{step}] {observation} -> {decision}", flush=True)


def parse_tapes(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        out[name.strip()] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--tape", action="append", default=[], help="dimension=metres, e.g. height=2.7")
    parser.add_argument("--instance-labels", type=Path, help="instance_labels.npy from semantics.run_sam")
    parser.add_argument("--no-reuse", action="store_true", help="recompute cached reconstructions")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log, gates = Log(), []

    # 1. Reconstruct with each candidate camera model and measure it.
    candidates = {}
    for name in args.models:
        info = reconstruct(args.images, args.out / name.lower(), name, reuse=not args.no_reuse)
        reg = gate_registration(info["n_registered"], info["n_images"], info["n_models"] or 1)
        info["registered_ok"] = reg.status == "pass"
        if info["registered_ok"]:
            pts, _, _ = read_model_points(Path(info["path"]))
            poses = read_model_poses(Path(info["path"]))
            frame, planes, labels, conf = analyse(pts, poses)
            on = labels != "object"
            info["plane_conf"] = float(conf[on].mean()) if on.any() else 0.0
        else:
            info["plane_conf"] = 0.0
        candidates[name] = info
        log.add(f"reconstruct:{name}", f"{info['n_registered']}/{info['n_images']} images, "
                f"{info['n_points']} points, reprojection {info['reproj_px']:.3f}px, "
                f"plane confidence {info['plane_conf']:.3f}; {reg.message}",
                "usable" if info["registered_ok"] else "discarded")

    # 2. Choose the camera model by comparison, not by an absolute threshold.
    choice = choose_model(candidates)
    for line in choice.log:
        log.add("model_choice", line, "")
    log.add("model_choice", f"{len(candidates)} candidates compared", f"use {choice.chosen}")
    chosen = candidates[choice.chosen]
    model_dir = Path(chosen["path"])
    gates.append(gate_registration(chosen["n_registered"], chosen["n_images"], chosen["n_models"] or 1))

    # 3. Structure and the floor plausibility gate.
    pts, _, _ = read_model_points(model_dir)
    poses = read_model_poses(model_dir)
    frame, planes, labels, conf = analyse(pts, poses)
    coords = pts @ frame.axes
    has_floor = any(p.role == "floor" for p in planes)
    cov = floor_coverage(coords, labels, frame.up) if has_floor else None
    floor_gate = gate_floor(cov)
    gates.append(floor_gate)
    log.add("floor_gate", floor_gate.message, floor_gate.status)

    # 4. Metric scale from tape measurements, cross-checked.
    dims = canonical_dimensions(dimensions(planes, frame))
    tapes = parse_tapes(args.tape)
    unknown = [k for k in tapes if k not in dims]
    for k in unknown:
        log.add("tape", f"dimension '{k}' not found in the reconstruction (found: {sorted(dims)})", "ignored")
    scales = {k: v / dims[k] for k, v in tapes.items() if k in dims}
    scale, scale_basis = None, "none"
    if scales:
        consistency = gate_tape_consistency(scales)
        gates.append(consistency)
        log.add("tape_consistency", consistency.message, consistency.status)
        usable = dict(scales)
        if consistency.status == "fail" and floor_gate.status == "warn" and "height" in usable:
            del usable["height"]
            log.add("tape_selection", "scales disagree and the floor gate warns, so the height tape is "
                    "the likely outlier", "drop 'height', keep the remaining tape(s)")
        if consistency.status == "fail" and len(usable) == len(scales):
            log.add("tape_selection", "scales disagree and no tape can be singled out", "no metric scale; ask the user to re-measure")
        elif usable:
            if "height" in usable and floor_gate.status == "warn" and len(usable) == 1:
                log.add("tape_selection", "only a height tape remains and the floor is doubtful", "no metric scale")
            else:
                scale = float(np.mean(list(usable.values())))
                scale_basis = "cross-checked" if len(usable) > 1 else "single tape (cannot be cross-checked)"
                log.add("scale", f"{len(usable)} tape(s) used: {sorted(usable)}", f"{scale:.5f} m per unit, {scale_basis}")
    else:
        log.add("scale", "no usable tape measurement", "dimensions stay in reconstruction units")

    # 5. Instance gate, if a SAM run is available.
    excluded = []
    if args.instance_labels:
        inst = np.load(args.instance_labels)
        if len(inst) != len(pts):
            log.add("instance_gate", f"instance labels cover {len(inst)} points but the chosen "
                    f"reconstruction has {len(pts)}: they come from a different model", "skipped; rerun SAM on this model")
        else:
            ext = np.percentile(coords, 99.5, axis=0) - np.percentile(coords, 0.5, axis=0)
            ig, excluded = gate_instance_extent(inst, coords, ext)
            gates.append(ig)
            log.add("instance_gate", ig.message, ig.status)

    # 6. Scene graph, with units stated.
    unit = "m" if scale else "units"
    graph = build_scene_graph(pts, labels, conf, frame, planes, scale=scale or 1.0, unit=unit)
    report = {
        "chosen_model": choice.chosen,
        "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in candidates.items()},
        "gates": [asdict(g) for g in gates],
        "metric_scale_m_per_unit": scale,
        "metric_scale_basis": scale_basis,
        "dimensions_units": dims,
        "excluded_instances": excluded,
        "decision_log": log.entries,
        "scene_graph": graph,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
