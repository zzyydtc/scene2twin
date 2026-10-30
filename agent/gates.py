"""Quality gates that need no reference scan.

Each gate encodes a failure met while building this project, and checks it from the
reconstruction itself, so the same checks can run on a room that has no ground truth.
A gate returns ``pass``, ``warn`` or ``fail`` with the value and threshold it used.
"""
from dataclasses import dataclass, field

import numpy as np

MODEL_ORDER = ["SIMPLE_RADIAL", "OPENCV", "OPENCV_FISHEYE"]  # simple to rich


@dataclass
class Gate:
    name: str
    status: str  # "pass" | "warn" | "fail"
    value: float | None
    threshold: float | None
    message: str


def gate_registration(n_registered: int, n_images: int, n_models: int, min_frac: float = 0.9) -> Gate:
    """Most images must sit in one model. Mapping can split a scene into fragments."""
    frac = n_registered / max(n_images, 1)
    if frac >= min_frac:
        note = "" if n_models == 1 else f" ({n_models} models produced; the largest is used)"
        return Gate("registration", "pass", frac, min_frac, f"{n_registered}/{n_images} images registered{note}")
    return Gate("registration", "fail", frac, min_frac,
                f"only {n_registered}/{n_images} images registered in the largest model")


def floor_coverage(coords: np.ndarray, labels: np.ndarray, up: int, grid: int = 12) -> float:
    """Share of occupied horizontal cells that contain floor-labelled points.

    A real floor spans the room; a desk top or other surface mistaken for it covers
    only part of it.
    """
    hz = [a for a in range(3) if a != up]
    lo = coords[:, hz].min(axis=0)
    cell = (coords[:, hz].max(axis=0) - lo).max() / grid
    if cell <= 0:
        return 0.0
    ij = np.floor((coords[:, hz] - lo) / cell).astype(int)
    key = ij[:, 0] * 10000 + ij[:, 1]
    occupied = len(set(key.tolist()))
    floor_cells = len(set(key[labels == "floor"].tolist()))
    return floor_cells / max(occupied, 1)


def gate_floor(coverage: float | None, min_cov: float = 0.5) -> Gate:
    if coverage is None:
        return Gate("floor_plausibility", "warn", None, min_cov,
                    "no floor plane observed; do not use height as a tape reference")
    if coverage >= min_cov:
        return Gate("floor_plausibility", "pass", coverage, min_cov, "floor plane spans the room")
    return Gate("floor_plausibility", "warn", coverage, min_cov,
                f"floor points cover {coverage:.0%} of the occupied floor area; the plane may be furniture, "
                "so floor-to-ceiling height is not a trustworthy tape reference")


def gate_tape_consistency(scales: dict[str, float], tol: float = 0.05) -> Gate:
    """Independent tape measurements must imply the same metric scale."""
    if len(scales) < 2:
        return Gate("tape_consistency", "warn", None, tol,
                    "fewer than two tape measurements: the scale cannot be cross-checked")
    vals = np.array(list(scales.values()))
    spread = float(vals.max() / vals.min() - 1.0)
    if spread <= tol:
        return Gate("tape_consistency", "pass", spread, tol, f"scales agree within {spread:.1%}")
    detail = ", ".join(f"{k}={v:.4f}" for k, v in scales.items())
    return Gate("tape_consistency", "fail", spread, tol, f"scales disagree by {spread:.1%} ({detail})")


def canonical_dimensions(dims: dict[str, float]) -> dict[str, float]:
    """Stable, user-facing names for measured room dimensions.

    Axis indices in a reconstruction are arbitrary and change between runs, so the
    horizontal dimensions are renamed: ``horizontal`` when one wall pair is observed,
    ``horizontal_long`` / ``horizontal_short`` when two are. ``height`` is unchanged.
    """
    out = {"height": dims["height"]} if "height" in dims else {}
    horiz = sorted((v for k, v in dims.items() if k != "height"), reverse=True)
    if len(horiz) == 1:
        out["horizontal"] = horiz[0]
    elif len(horiz) >= 2:
        out["horizontal_long"], out["horizontal_short"] = horiz[0], horiz[-1]
    return out


def gate_instance_extent(
    instance_labels: np.ndarray, coords: np.ndarray, room_extent: np.ndarray, frac: float = 0.5,
    min_axes: int = 2,
) -> tuple[Gate, list[int]]:
    """Instances as large as the room in several axes are chained merges, not objects."""
    bad = []
    for k in sorted(set(instance_labels[instance_labels >= 0].tolist())):
        c = coords[instance_labels == k]
        big = (np.ptp(c, axis=0) > frac * room_extent).sum()
        if big >= min_axes:
            bad.append(k)
    if not bad:
        return Gate("instance_extent", "pass", 0, frac, "no room-sized instances"), bad
    return Gate("instance_extent", "warn", len(bad), frac,
                f"{len(bad)} instance(s) span more than {frac:.0%} of the room in {min_axes}+ axes "
                f"(ids {bad}); excluded as likely chained merges"), bad


@dataclass
class ModelChoice:
    chosen: str
    log: list[str] = field(default_factory=list)


def choose_model(candidates: dict[str, dict], min_improvement: float = 0.2, max_quality_loss: float = 0.02) -> ModelChoice:
    """Pick a camera model from candidate reconstructions.

    ``candidates[name]`` needs ``registered_ok`` (bool), ``reproj_px`` and ``plane_conf``
    (mean confidence of points on detected planes). Absolute reprojection error is not a
    usable threshold (a wrong model can still be under 1 px), so models are compared:
    walking from the simplest, a richer model replaces the current choice only if it
    lowers the reprojection error by at least ``min_improvement`` (richer models always
    fit somewhat better) without lowering plane quality by more than ``max_quality_loss``.
    """
    ordered = [m for m in MODEL_ORDER if m in candidates and candidates[m]["registered_ok"]]
    if not ordered:
        raise RuntimeError("no candidate model passed the registration gate")
    choice = ModelChoice(ordered[0], [f"start with {ordered[0]}"])
    for name in ordered[1:]:
        cur, new = candidates[choice.chosen], candidates[name]
        gain = 1.0 - new["reproj_px"] / cur["reproj_px"]
        loss = cur["plane_conf"] - new["plane_conf"]
        if gain >= min_improvement and loss <= max_quality_loss:
            choice.log.append(f"{name} replaces {choice.chosen}: reprojection {gain:+.0%} better, "
                              f"plane confidence change {-loss:+.3f}")
            choice.chosen = name
        else:
            choice.log.append(f"{name} rejected vs {choice.chosen}: reprojection gain {gain:+.0%} "
                              f"(needs {min_improvement:.0%}), plane confidence change {-loss:+.3f}")
    return choice
