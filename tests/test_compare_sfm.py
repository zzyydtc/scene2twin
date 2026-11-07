import numpy as np
import pytest

from eval.compare_sfm import run_engine, summarise


def _run(engine, seconds, ate_m, registered=26):
    return {
        "engine": engine, "seconds": seconds, "n_registered": registered, "n_points": 1000,
        "poses": {"ate_rmse_m": ate_m, "rotation_err_median_deg": 0.4},
        "accuracy_median_m": 0.008, "precision_2cm": 0.95,
        "reprojection": {"reproj_mean_px": 0.41},
    }


def test_summarise_groups_by_engine_and_converts_units():
    runs = [_run("incremental", 10, 0.010), _run("incremental", 14, 0.020), _run("global", 5, 0.015)]
    s = summarise(runs)
    assert set(s) == {"incremental", "global"}
    assert s["incremental"]["n_scored"] == 2
    assert s["incremental"]["seconds"]["mean"] == pytest.approx(12.0)
    assert s["incremental"]["camera_rmse_cm"]["mean"] == pytest.approx(1.5)
    assert s["incremental"]["camera_rmse_cm"]["max"] == pytest.approx(2.0)
    assert s["global"]["point_median_mm"]["mean"] == pytest.approx(8.0)
    assert s["global"]["camera_rmse_cm"]["std"] == pytest.approx(0.0)


def test_summarise_leaves_failed_runs_out_of_the_statistics_but_counts_them():
    failed = {"engine": "global", "seconds": 3.0, "n_models": 0, "model": None}
    s = summarise([_run("global", 5, 0.01), failed])
    assert s["global"]["n_runs"] == 2
    assert s["global"]["n_scored"] == 1
    assert s["global"]["seconds"]["mean"] == pytest.approx(5.0)


def test_summarise_with_no_scored_runs_reports_counts_only():
    s = summarise([{"engine": "global", "seconds": 3.0, "n_models": 0, "model": None}])
    assert s["global"] == {"n_runs": 1, "n_scored": 0}


def test_run_engine_rejects_an_unknown_engine(tmp_path):
    db = tmp_path / "db.db"
    db.write_bytes(b"")
    with pytest.raises(ValueError):
        run_engine("sift_but_better", db, tmp_path, tmp_path / "out", 1)
