"""Machine-specific paths live in ignored local.json, never in source control."""

import json
import os
from pathlib import Path


def load_config(repo: Path) -> dict:
    path = repo / "local.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data_root = os.environ.get("SIC_XRT_DATA_ROOT", values.get("data_root"))
    port = values.get("port", 8890)
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("port must be an integer between 1024 and 65535")
    if not data_root:
        raise ValueError("Set data_root in local.json or SIC_XRT_DATA_ROOT")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Data folder does not exist: {root}")
    resolved_repo = repo.resolve()
    # Work, runtime and reports must never be created beneath the original data.
    if resolved_repo == root or root in resolved_repo.parents:
        raise ValueError("Keep the code checkout outside the original data folder")
    return {"data_root": str(root), "port": port}
