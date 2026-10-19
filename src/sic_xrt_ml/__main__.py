"""Explicit CLI stages; no automatic training, export, deployment or uploads."""

import argparse
from pathlib import Path

from .data.verified import read_json


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("dataset")
    p = sub.add_parser("train")
    p.add_argument("dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--synthetic-smoke", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("dataset")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--existing-bundle")
    p.add_argument("--synthetic-smoke", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--output", required=True)
    a = parser.parse_args()
    if a.command == "validate":
        from .data.verified import readiness, validate

        result = readiness(validate(a.dataset))
    elif a.command == "train":
        from .training.engine import TrainConfig, train

        result = train(
            a.dataset,
            a.output,
            TrainConfig(**read_json(Path(a.config).read_bytes())),
            a.resume,
            a.synthetic_smoke,
        )
    elif a.command == "evaluate":
        from .evaluation.compare import evaluate

        result = evaluate(
            a.dataset, a.checkpoint, a.output, a.existing_bundle, a.synthetic_smoke
        )
    else:
        from .export.onnx_bundle import export

        result = export(a.checkpoint, a.output, a.evaluation)
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
