"""Manage this checkout's authenticated, loopback-only JupyterLab server."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from .config import load_config


def runtime_environment(repo: Path, settings: dict) -> dict:
    env = os.environ.copy()
    env.update({
        "SIC_XRT_DATA_ROOT": settings["data_root"],
        "SIC_XRT_REPO": str(repo),
        "JUPYTER_RUNTIME_DIR": str(repo / "outputs" / "runtime"),
        "JUPYTER_CONFIG_DIR": str(repo / "outputs" / "jupyter-config"),
        "JUPYTER_DATA_DIR": str(repo / "outputs" / "jupyter-data"),
        "JUPYTER_PATH": str(Path(sys.prefix) / "share" / "jupyter"),
    })
    # Do not inherit environment settings that turn off default authentication.
    env.pop("JUPYTER_TOKEN", None)
    env.pop("JUPYTER_TOKEN_FILE", None)
    return env


def server_infos(repo: Path, port: int):
    for path in (repo / "outputs" / "runtime").glob("jpserver-*.json"):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            if (info.get("port") == port
                    and Path(info.get("root_dir", "")).resolve() == (repo / "work").resolve()
                    and info.get("token")):
                yield info
        except (OSError, ValueError):
            continue


def api_request(port: int, info: dict, endpoint: str, method: str = "GET"):
    request = Request(
        f"http://127.0.0.1:{port}/api/{endpoint}",
        headers={"Authorization": f"token {info['token']}"},
        data=b"" if method == "POST" else None,
        method=method,
    )
    # Local server requests should not go through a machine-wide proxy.
    with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
        return response.status


def live_info(repo: Path, port: int) -> dict | None:
    for info in server_infos(repo, port):
        try:
            if api_request(port, info, "status") == 200:
                return info
        except (OSError, URLError):
            pass
    return None


def launch(repo: Path, settings: dict) -> dict:
    port = settings["port"]
    existing = live_info(repo, port)
    if existing:
        return existing
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"Port {port} is in use; change local.json port") from error
    env = runtime_environment(repo, settings)
    for key in ("JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    work = repo / "work"
    work.mkdir(exist_ok=True)
    template = repo / "notebooks" / "00_environment.ipynb"
    notebook = work / template.name
    if not notebook.exists():
        shutil.copyfile(template, notebook)
    command = [
        sys.executable, "-m", "jupyterlab",
        "--ServerApp.ip=127.0.0.1", f"--ServerApp.port={port}",
        "--ServerApp.port_retries=0", "--ServerApp.open_browser=False",
        f"--ServerApp.root_dir={work}",
    ]
    options = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    # Jupyter logs may contain a token; the entire outputs/ tree is ignored.
    with (repo / "outputs" / "server.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, **options)
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Jupyter exited; inspect outputs/server.log locally")
        info = live_info(repo, port)
        if info:
            return info
        time.sleep(0.25)
    # Stop only the child we just started, rather than leaving an orphan.
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError("Jupyter did not become ready within 40 seconds")


def stop(repo: Path, settings: dict) -> None:
    info = live_info(repo, settings["port"])
    if not info:
        print("This checkout's server is already stopped.")
        return
    api_request(settings["port"], info, "shutdown", method="POST")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if live_info(repo, settings["port"]) is None:
            print("Server stopped.")
            return
        time.sleep(0.25)
    raise RuntimeError("Shutdown not confirmed; check the server locally")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "open", "status", "stop", "restart"])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        settings = load_config(repo)
        if args.action == "status":
            running = live_info(repo, settings["port"]) is not None
            print(f"{'Running' if running else 'Stopped'}: http://127.0.0.1:{settings['port']}/lab")
            return 0 if running else 1
        if args.action in {"stop", "restart"}:
            stop(repo, settings)
            if args.action == "stop":
                return 0
        info = launch(repo, settings)
        print(f"Ready: http://127.0.0.1:{settings['port']}/lab")
        if not args.no_browser:
            url = f"http://127.0.0.1:{settings['port']}/lab?{urlencode({'token': info['token']})}"
            webbrowser.open(url)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
