import numpy as np
import pytest

from agent.gates import (
    canonical_dimensions, choose_model, floor_coverage, gate_floor, gate_instance_extent, gate_registration,
    gate_tape_consistency,
)


def test_registration_gate_pass_and_fail():
    assert gate_registration(26, 26, 1).status == "pass"
    assert "2 models" in gate_registration(26, 26, 2).message
    assert gate_registration(7, 26, 2).status == "fail"


def test_floor_coverage_distinguishes_room_floor_from_desk():
    rng = np.random.default_rng(0)
    # Horizontal axes 0 and 1, vertical axis 2. Points everywhere (walls etc.).
    everywhere = np.column_stack([rng.uniform(0, 10, 4000), rng.uniform(0, 10, 4000), rng.uniform(0, 3, 4000)])
    labels_full = np.where(everywhere[:, 2] < 0.05, "floor", "object")
    full_floor = np.column_stack([rng.uniform(0, 10, 2000), rng.uniform(0, 10, 2000), np.zeros(2000)])
    desk = np.column_stack([rng.uniform(0, 3, 2000), rng.uniform(0, 3, 2000), np.zeros(2000)])
    for floor_pts, expect_high in ((full_floor, True), (desk, False)):
        coords = np.vstack([everywhere, floor_pts])
        labels = np.concatenate([np.full(len(everywhere), "object"), np.full(len(floor_pts), "floor")])
        cov = floor_coverage(coords, labels, up=2)
        assert (cov > 0.7) == expect_high
    assert gate_floor(0.3).status == "warn" and gate_floor(0.8).status == "pass"
    assert gate_floor(None).status == "warn"


def test_tape_consistency():
    assert gate_tape_consistency({"height": 0.1867, "horizontal_1": 0.1566}).status == "fail"
    assert gate_tape_consistency({"a": 0.1500, "b": 0.1510}).status == "pass"
    assert gate_tape_consistency({"a": 0.15}).status == "warn"


def test_instance_extent_flags_room_sized_instances():
    rng = np.random.default_rng(1)
    room = np.array([10.0, 8.0, 3.0])
    coords = np.vstack([rng.uniform(0, 1, (50, 3)), rng.uniform(0, 1, (50, 3)) + 5, rng.uniform(0, 1, (10, 3))])
    labels = np.array([0] * 50 + [1] * 50 + [2] * 10)
    coords[labels == 2] = rng.uniform(0, 1, (10, 3)) * room  # spans the whole room
    gate, bad = gate_instance_extent(labels, coords, room)
    assert bad == [2] and gate.status == "warn"
    assert gate_instance_extent(labels[:100], coords[:100], room)[0].status == "pass"


def test_choose_model_requires_clear_improvement():
    cands = {
        "SIMPLE_RADIAL": {"registered_ok": True, "reproj_px": 0.98, "plane_conf": 0.78},
        "OPENCV": {"registered_ok": True, "reproj_px": 0.93, "plane_conf": 0.78},
        "OPENCV_FISHEYE": {"registered_ok": True, "reproj_px": 0.45, "plane_conf": 0.89},
    }
    choice = choose_model(cands)
    assert choice.chosen == "OPENCV_FISHEYE"
    assert any("OPENCV rejected" in line for line in choice.log)


def test_choose_model_rejects_richer_model_with_worse_planes():
    cands = {
        "SIMPLE_RADIAL": {"registered_ok": True, "reproj_px": 1.0, "plane_conf": 0.9},
        "OPENCV_FISHEYE": {"registered_ok": True, "reproj_px": 0.4, "plane_conf": 0.7},
    }
    assert choose_model(cands).chosen == "SIMPLE_RADIAL"


def test_choose_model_skips_models_that_failed_registration():
    cands = {
        "SIMPLE_RADIAL": {"registered_ok": True, "reproj_px": 1.0, "plane_conf": 0.8},
        "OPENCV_FISHEYE": {"registered_ok": False, "reproj_px": 0.2, "plane_conf": 0.9},
    }
    assert choose_model(cands).chosen == "SIMPLE_RADIAL"
    with pytest.raises(RuntimeError):
        choose_model({"OPENCV": {"registered_ok": False, "reproj_px": 1, "plane_conf": 1}})


def test_canonical_dimension_names_ignore_axis_indices():
    assert canonical_dimensions({"height": 21.0, "horizontal_2": 19.9}) == {"height": 21.0, "horizontal": 19.9}
    assert canonical_dimensions({"horizontal_0": 15.0, "horizontal_1": 20.0}) == {
        "horizontal_long": 20.0, "horizontal_short": 15.0}
    assert canonical_dimensions({"height": 2.0}) == {"height": 2.0}
