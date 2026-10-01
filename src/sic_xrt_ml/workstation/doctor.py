"""Actual CUDA forward/backward smoke check, not a defect model evaluation."""

import argparse
import json
import platform
import sys


def diagnose(require_cuda: bool = False) -> dict:
    import torch

    available = torch.cuda.is_available()
    result = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "purpose": "environment_smoke_check_only",
    }
    if require_cuda and not available:
        raise RuntimeError("CUDA is unavailable; GPU environment validation failed")
    device = torch.device("cuda" if available else "cpu")
    if available:
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu=properties.name, vram_bytes=properties.total_memory)
    # Small tensor check exercises convolution kernels and autograd on the GPU.
    layer = torch.nn.Conv2d(1, 4, 3, padding=1).to(device)
    image = torch.ones((2, 1, 128, 128), device=device)
    loss = layer(image).square().mean()
    loss.backward()
    if available:
        torch.cuda.synchronize()
    gradient = layer.weight.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("Forward/backward check produced invalid gradients")
    result.update(device=str(device), forward_backward="passed", loss=float(loss.detach()))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.require_cuda), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
