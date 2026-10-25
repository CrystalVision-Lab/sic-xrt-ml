"""Research-only frozen v1 bundle with an extra subtype output.

``probabilities`` [N,3] keeps the exact frozen_xrt_patch_classifier v1 contract (sum
of fine probabilities per class), so v1 consumers that run ``["probabilities"]`` keep
working. ``subtype_probabilities`` [N,10] is described by the new ``subtype_output``
manifest key. ``coarse_only=True`` exports the same model with the single
``probabilities`` output, for consumers that require exactly one ONNX output (the
Analyzer model-update/registry validation at the time of #51). No replacement or
distribution of any existing model.
"""

import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch

from ..data.provider_subtype import CLASSES, FINE_LABELS, FINE_TO_CLASS
from ..data.verified import canonical, digest, read_json
from ..evaluation.compare import INPUT, OUTPUT, SUBTYPE_OUTPUT, frozen_session
from ..training.subtype import load_checkpoint, selected_model
from ..training.subtype_model import SubtypeScoreExport

LIMITATIONS = [
    "provider_labels_not_expert_ground_truth",
    "classification_of_candidate_patches_only_no_background_class",
    "direction_subtypes_valid_only_on_tiff_pixel_orientation_mirrored_input_mirrors_subtypes",
    "provider_jpegs_are_horizontally_mirrored_copies_of_the_tiffs",
    "rare_subtypes_TED_e_TED_f_TSD_c_have_few_training_samples",
    "trained_on_two_areas_only_scores_uncalibrated",
]
ATOL, RTOL = 1e-5, 1e-4


def export_subtype(checkpoint_path, output, evaluation_path, coarse_only=False):
    checkpoint = load_checkpoint(checkpoint_path)
    evaluation = read_json(Path(evaluation_path).read_bytes())
    sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    if (
        evaluation.get("task") != "provider_subtype"
        or evaluation["checkpoint_sha256"] != sha
        or evaluation["dataset_hash"] != checkpoint["dataset_hash"]
    ):
        raise ValueError("EVALUATION_PROVENANCE")
    if evaluation["scientific_metrics"] == checkpoint["synthetic_smoke"]:
        raise ValueError("EVALUATION_ORIGIN")
    if (
        evaluation["fine_labels"] != FINE_LABELS
        or checkpoint["fine_labels"] != FINE_LABELS
        or evaluation["classes"] != CLASSES
    ):
        raise ValueError("EVALUATION_CLASSES")
    out = Path(output)
    if out.exists():
        raise ValueError("NEW_CANDIDATE_DIRECTORY_REQUIRED")
    out.mkdir(parents=True, mode=0o700)
    torch.set_num_threads(2)
    model = SubtypeScoreExport(selected_model(checkpoint), coarse_only).eval()
    rng = np.random.default_rng(2026)
    example = torch.from_numpy(rng.random((1, 3, 128, 128), dtype=np.float32))
    path = out / "model.onnx"
    names = [OUTPUT["name"]] + ([] if coarse_only else [SUBTYPE_OUTPUT["name"]])
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["images"],
        output_names=names,
        dynamic_axes={"images": {0: "N"}, **{n: {0: "N"} for n in names}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    model_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "frozen_xrt_patch_classifier",
        "schema_version": 1,
        "status": "validated_export",
        "classes": list(CLASSES),
        "input": INPUT,
        "output": OUTPUT,
        **({} if coarse_only else {"subtype_output": SUBTYPE_OUTPUT}),
        "trained_fine_labels": FINE_LABELS,
        "coarse_rule": "probabilities = per-class sums of the fine-label softmax",
        "model_file": "model.onnx",
        "model_sha256": model_sha,
        "bundle_id": ("provider-class-" if coarse_only else "provider-subtype-")
        + model_sha[:12],
        "research_only": True,
        "independent_test_passed": False,
        "background_class": False,
        "localizes_defects": False,
        "labels_source": "provider",
        "expert_verified": False,
        "limitations": LIMITATIONS,
        "architecture": checkpoint["architecture"],
        "dataset_hash": checkpoint["dataset_hash"],
        "dataset": checkpoint["dataset"],
        "config_hash": checkpoint["config_hash"],
        "code_revision": checkpoint["code_revision"],
        "checkpoint_sha256": sha,
        "evaluation_hash": digest(evaluation),
        "synthetic_smoke": checkpoint["synthetic_smoke"],
        "improvement": "NOT_VERIFIED",
        "candidate_status": "FUNCTIONAL_TEST_ONLY"
        if checkpoint["synthetic_smoke"]
        else "AWAITING_EXPERT_MODEL_ACCEPTANCE",
    }
    (out / "manifest.json").write_bytes(canonical(manifest))
    try:
        session, _ = frozen_session(out)
        groups = np.asarray(FINE_TO_CLASS)
        parity = []
        for batch in (1, 7):
            x = rng.random((batch, 3, 128, 128), dtype=np.float32)
            with torch.no_grad():
                full = SubtypeScoreExport(model.model)(torch.from_numpy(x))
            expected = [t.numpy() for t in full][: len(names)]
            actual = session.run(names, {"images": x})
            coarse = actual[0]
            # The coarse-only graph is checked against the same model's fine scores.
            fine = actual[1] if len(actual) > 1 else full[1].numpy()
            summed = np.stack([fine[:, groups == k].sum(1) for k in range(3)], 1)
            if (
                coarse.shape != (batch, 3)
                or fine.shape != (batch, len(FINE_LABELS))
                or not all(np.isfinite(a).all() and (a >= 0).all() for a in actual)
                or not np.allclose(coarse.sum(1), 1, atol=1e-4)
                or not np.allclose(fine.sum(1), 1, atol=1e-4)
                or not np.allclose(coarse, summed, atol=1e-5)
                or not all(
                    np.allclose(a, e, atol=ATOL, rtol=RTOL)
                    for a, e in zip(actual, expected, strict=True)
                )
            ):
                raise ValueError("ONNX_RUNTIME_PARITY")
            parity.append(
                {
                    "batch": batch,
                    **{
                        f"{n}_max_absolute_error": float(np.abs(a - e).max())
                        for n, a, e in zip(names, actual, expected, strict=True)
                    },
                    "coarse_equals_group_sum_max_error": float(np.abs(coarse - summed).max()),
                }
            )
        manifest["parity"] = {
            "provider": "CPUExecutionProvider",
            "atol": ATOL,
            "rtol": RTOL,
            "outputs": names,
            "results": parity,
        }
        (out / "manifest.json").write_bytes(canonical(manifest))
        registry = {
            "schema": "xrt_model_candidate/v1",
            "candidate_id": model_sha,
            "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
            "dataset_hash": checkpoint["dataset_hash"],
            "dataset": checkpoint["dataset"],
            "evaluation": evaluation,
            "status": manifest["candidate_status"],
            "distribution": "NOT_AUTHORIZED",
            "replaces_existing_model": False,
        }
        (out / "candidate-registry.json").write_bytes(canonical(registry))
    except Exception:
        (out / "manifest.json").unlink(missing_ok=True)
        raise
    return manifest
