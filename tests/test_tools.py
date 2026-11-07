import pycolmap

from agent import tools


def _capture(monkeypatch):
    seen = {}
    monkeypatch.setattr(pycolmap, "extract_features", lambda *a, **k: None)
    monkeypatch.setattr(pycolmap, "match_exhaustive", lambda *a, **k: None)

    def fake_mapping(database, images, sparse, options):
        seen["options"] = options
        return {}

    monkeypatch.setattr(pycolmap, "incremental_mapping", fake_mapping)
    return seen


def _images(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"")
    return folder


def test_principal_point_stays_fixed_by_default(tmp_path, monkeypatch):
    seen = _capture(monkeypatch)
    info = tools.reconstruct(_images(tmp_path), tmp_path / "out", "OPENCV")
    assert seen["options"].ba_refine_principal_point is False
    assert info["n_registered"] == 0


def test_principal_point_can_be_refined(tmp_path, monkeypatch):
    seen = _capture(monkeypatch)
    tools.reconstruct(_images(tmp_path), tmp_path / "out", "OPENCV", refine_principal_point=True)
    assert seen["options"].ba_refine_principal_point is True
