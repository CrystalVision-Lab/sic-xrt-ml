"""Explicit CLI stages; no automatic training, export, deployment or uploads."""

import argparse
from pathlib import Path

from .data.verified import read_json


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gate = (
        "v2 / provider subtype only: minimum usable samples per class as "
        "train,validation,test (default 10,5,5; synthetic smoke 1,1,1)"
    )
    p = sub.add_parser("validate")
    p.add_argument("dataset")
    p.add_argument("--min-per-class", help=gate)
    p.add_argument("--synthetic-smoke", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--min-per-class", help=gate)
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
    p.add_argument(
        "--coarse-only",
        action="store_true",
        help="provider subtype only: single probabilities [N,3] output",
    )
    p = sub.add_parser("crossval", help="provider subtype only: leave-one-Area-out")
    p.add_argument("dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--existing-bundle")
    p.add_argument("--synthetic-smoke", action="store_true")
    a = parser.parse_args()
    if a.command == "export":
        from .training.subtype import is_subtype_checkpoint

        subtype = is_subtype_checkpoint(a.checkpoint)
    else:
        from .data.loading import is_provider_subtype

        subtype = is_provider_subtype(a.dataset)
    if a.command == "crossval" and not subtype:
        raise ValueError("CROSSVAL_PROVIDER_SUBTYPE_ONLY")
    if subtype:
        result = subtype_command(a)
    elif a.command == "validate":
        from .data.loading import parse_minimum, prepare

        prepared = prepare(
            a.dataset, a.synthetic_smoke, parse_minimum(a.min_per_class)
        )
        result = {**prepared.readiness, "dataset": prepared.provenance()}
    elif a.command == "train":
        from .data.loading import parse_minimum
        from .training.engine import TrainConfig, train

        result = train(
            a.dataset,
            a.output,
            TrainConfig(**read_json(Path(a.config).read_bytes())),
            a.resume,
            a.synthetic_smoke,
            parse_minimum(a.min_per_class),
        )
    elif a.command == "evaluate":
        from .evaluation.compare import evaluate

        result = evaluate(
            a.dataset, a.checkpoint, a.output, a.existing_bundle, a.synthetic_smoke
        )
    else:
        if a.coarse_only:
            raise ValueError("COARSE_ONLY_PROVIDER_SUBTYPE_ONLY")
        from .export.onnx_bundle import export

        result = export(a.checkpoint, a.output, a.evaluation)
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


def subtype_command(a):
    """provider_subtype_point_dataset/v1: the parallel fine-label pipeline."""
    from .data import provider_subtype as ps

    if a.command == "export":
        from .export.subtype_bundle import export_subtype

        return export_subtype(a.checkpoint, a.output, a.evaluation, a.coarse_only)
    if a.command == "evaluate":
        from .evaluation.subtype_eval import evaluate_subtype

        return evaluate_subtype(
            a.dataset, a.checkpoint, a.output, a.existing_bundle, a.synthetic_smoke
        )
    from .training.subtype import SubtypeTrainConfig

    minimum = None
    if getattr(a, "min_per_class", None):
        parts = a.min_per_class.split(",")
        if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
            raise ValueError("MIN_PER_CLASS")
        minimum = ps.minimum_gate(dict(zip(ps.SPLITS, map(int, parts), strict=True)))
    if a.command == "validate":
        data = ps.validate(a.dataset)
        assignment = ps.split_assignment(data)
        return {
            **ps.readiness(data, assignment, a.synthetic_smoke, minimum),
            "dataset": data.provenance(),
        }
    config = SubtypeTrainConfig(**read_json(Path(a.config).read_bytes()))
    if a.command == "train":
        from .training.subtype import train_subtype

        return train_subtype(
            a.dataset, a.output, config, a.resume, a.synthetic_smoke, minimum=minimum
        )
    from .evaluation.subtype_eval import cross_validate

    return cross_validate(
        a.dataset, a.output, config, a.existing_bundle, a.synthetic_smoke
    )


if __name__ == "__main__":
    main()
