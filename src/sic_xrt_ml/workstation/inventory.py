"""Inspect external file metadata without extracting ZIPs or decoding TIFF pixels."""

import argparse
import json
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config


def inspect_data(root: Path) -> dict:
    import tifffile

    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Data folder does not exist: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Never follow a symlink to an unrelated folder during inspection.
        if root not in path.resolve().parents:
            continue
        stat = path.stat()
        entry = {
            "path": path.relative_to(root).as_posix(),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    members = [m for m in archive.infolist() if not m.is_dir()]
                    entry["zip"] = {
                        "files": len(members),
                        "extensions": dict(Counter(Path(m.filename).suffix.lower()
                                                   for m in members)),
                        "uncompressed_bytes": sum(m.file_size for m in members),
                        "roi_members": [m.filename for m in members
                                        if Path(m.filename).suffix.lower() == ".roi"],
                        "nested_archives": [m.filename for m in members
                                            if Path(m.filename).suffix.lower() == ".zip"],
                        "pixel_integrity_checked": False,
                    }
            elif path.suffix.lower() in {".tif", ".tiff"}:
                with tifffile.TiffFile(path) as image:
                    series = image.series[0]
                    page = image.pages[0]
                    entry["tiff"] = {
                        "pages": len(image.pages),
                        "shape": list(series.shape),
                        "axes": series.axes,
                        "dtype": str(page.dtype),
                        "imagej": image.is_imagej,
                        "compression": tifffile.COMPRESSION(page.compression).name,
                        "pixel_integrity_checked": False,
                    }
            if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
                entry["error"] = "File changed during inspection; rerun after copying finishes"
        except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        records.append(entry)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "data_root": str(root),
        "inspection": "metadata_only_no_extraction_no_pixel_decode",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "error_count": sum("error" in record for record in records),
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    settings = load_config(args.repo)
    report = inspect_data(Path(settings["data_root"]))
    output = args.repo.resolve() / "outputs" / "inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Inspected {report['file_count']} files; {report['error_count']} errors. Report: {output}")
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
