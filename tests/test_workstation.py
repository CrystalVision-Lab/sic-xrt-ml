import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from sic_xrt_ml.workstation.config import load_config
from sic_xrt_ml.workstation.inventory import inspect_data
from sic_xrt_ml.workstation.server import live_info, runtime_environment


def test_metadata_inspection_preserves_originals_and_reports_roi(tmp_path):
    root = tmp_path / "originals"
    root.mkdir()
    image = root / "stack.tif"
    tifffile.imwrite(image, np.zeros((3, 7, 8), dtype=np.uint16), imagej=True,
                     metadata={"axes": "TYX"})
    archive = root / "area.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("area/image.tif", b"not decoded")
        output.writestr("area/BPD.roi", b"not interpreted")
        output.writestr("area/nested.zip", b"not extracted")
    originals = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
    report = inspect_data(root)
    assert report["file_count"] == 2
    assert report["error_count"] == 0
    tiff = next(r["tiff"] for r in report["files"] if "tiff" in r)
    assert tiff["shape"] == [3, 7, 8]
    assert tiff["axes"] == "TYX"
    assert tiff["dtype"] == "uint16"
    assert not tiff["pixel_integrity_checked"]
    zipped = next(r["zip"] for r in report["files"] if "zip" in r)
    assert zipped["roi_members"] == ["area/BPD.roi"]
    assert zipped["nested_archives"] == ["area/nested.zip"]
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()} == originals


def test_corrupt_archive_reports_error_without_hiding_other_files(tmp_path):
    (tmp_path / "bad.zip").write_bytes(b"bad zip")
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    report = inspect_data(tmp_path)
    assert report["file_count"] == 2
    assert report["error_count"] == 1
    assert "error" in report["files"][0]


def test_missing_data_root_does_not_report_false_success(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        inspect_data(tmp_path / "missing")


@pytest.mark.parametrize("port", [True, "8890", 80, 65536])
def test_invalid_port_rejected(tmp_path, port):
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "local.json").write_text(json.dumps({"data_root": str(data), "port": port}))
    with pytest.raises(ValueError, match="port"):
        load_config(tmp_path)


def test_code_cannot_be_inside_originals(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "local.json").write_text(json.dumps({"data_root": str(tmp_path)}))
    with pytest.raises(ValueError, match="outside"):
        load_config(repo)


def test_environment_override_and_token_isolation(tmp_path, monkeypatch):
    data = tmp_path / "originals"
    data.mkdir()
    monkeypatch.setenv("SIC_XRT_DATA_ROOT", str(data))
    monkeypatch.setenv("JUPYTER_TOKEN", "")
    monkeypatch.setenv("JUPYTER_TOKEN_FILE", "somefile")
    settings = load_config(tmp_path)
    env = runtime_environment(tmp_path, settings)
    assert Path(env["SIC_XRT_DATA_ROOT"]) == data
    assert "JUPYTER_TOKEN" not in env
    assert "JUPYTER_TOKEN_FILE" not in env
    assert Path(env["JUPYTER_RUNTIME_DIR"]).is_relative_to(tmp_path)


def test_stale_runtime_does_not_hide_a_live_server(tmp_path, monkeypatch):
    runtime = tmp_path / "outputs" / "runtime"
    runtime.mkdir(parents=True)
    common = {"port": 8890, "root_dir": str(tmp_path / "work")}
    (runtime / "jpserver-1.json").write_text(json.dumps({**common, "token": "stale"}))
    (runtime / "jpserver-2.json").write_text(json.dumps({**common, "token": "live"}))

    def fake_request(port, info, endpoint):
        if info["token"] == "stale":
            raise OSError("stale process")
        return 200

    monkeypatch.setattr("sic_xrt_ml.workstation.server.api_request", fake_request)
    assert live_info(tmp_path, 8890)["token"] == "live"
