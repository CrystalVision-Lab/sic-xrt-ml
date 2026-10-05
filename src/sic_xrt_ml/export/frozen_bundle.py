"""Freeze a research spatial classifier into a self-contained, verified ONNX bundle."""

import copy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ..training.patch_classifier import CLASSES, file_hash, write_json
from ..training.recall_budget import read_json
from ..training.spatial_robustness import load_spatial_candidate


class ExportProbabilities(torch.nn.Module):
    def __init__(self, candidate):
        super().__init__()
        self.models = copy.deepcopy(candidate.model.members)
        for model in self.models:
            if model.center_size != 64:
                raise ValueError('This frozen export contract requires center64')
            # Both branches enter their final pooling at 8x8 for fixed RGB128 input.
            model.classifier.center.features[-1] = torch.nn.AvgPool2d(2)
            model.classifier.context[-1] = torch.nn.AvgPool2d(2)
        self.views = 4 if candidate.mode == 'c4' else 1

    def forward(self, images):
        # Primitive transpose/flip preserves rot90 semantics with portable ONNX ops.
        views = (images, images.transpose(-1, -2).flip(-2),
                 images.flip((-2, -1)), images.transpose(-1, -2).flip(-1))
        return torch.stack([model(view).softmax(1) for view in views[:self.views]
                            for model in self.models]).mean(0)


def freeze_bundle(pointer, output, probes, code_revision):
    import onnx
    import onnxruntime as ort

    pointer, output = Path(pointer).resolve(), Path(output).resolve()
    if output.exists() or not code_revision:
        raise ValueError('New output and explicit code revision required')
    if probes.dtype != np.uint8 or probes.ndim != 4 or probes.shape[1:] != (128, 128, 3) or len(probes) < 3:
        raise ValueError('At least three raw uint8 RGB128 development probes required')
    source = read_json(pointer)
    candidate = load_spatial_candidate(pointer, 'cpu')
    model = ExportProbabilities(candidate).eval()
    inputs = torch.from_numpy(probes.transpose(0, 3, 1, 2).copy()).float()/255
    with torch.inference_mode():
        expected = candidate(inputs).softmax(1).numpy()
        if not np.allclose(model(inputs).numpy(), expected, atol=1e-6):
            raise ValueError('Export wrapper changed probabilities')
    output.mkdir(parents=True)
    path = output/'model.onnx'
    torch.onnx.export(model, (inputs[:1],), str(path), dynamo=False, opset_version=17,
                      input_names=['images'], output_names=['probabilities'],
                      dynamic_axes={'images': {0: 'batch'}, 'probabilities': {0: 'batch'}})
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    comparisons = []
    for indices in ([0], list(range(min(7, len(inputs)))), list(range(len(inputs)))):
        observed = session.run(['probabilities'], {'images': inputs[indices].numpy()})[0]
        delta = float(np.max(np.abs(observed-expected[indices])))
        if not np.allclose(observed, expected[indices], atol=1e-4) or not np.array_equal(observed.argmax(1), expected[indices].argmax(1)):
            raise ValueError('ONNX probability/label parity failed')
        comparisons.append({'batch': len(indices), 'max_absolute_error': delta, 'labels_equal': True})
    manifest = {'schema': 'frozen_xrt_patch_classifier', 'schema_version': 1,
        'created_at': datetime.now(UTC).isoformat(), 'bundle_id': 'spatial64-c4-'+file_hash(path)[:12],
        'model_file': 'model.onnx', 'model_sha256': file_hash(path), 'classes': list(CLASSES),
        'input': {'name': 'images', 'dtype': 'float32', 'shape': ['N', 3, 128, 128],
                  'conversion': 'uint8_RGB_to_NCHW_div255', 'local_contrast_inside_model': True},
        'output': {'name': 'probabilities', 'shape': ['N', 3], 'meaning': 'uncalibrated_class_scores'},
        'source_pointer_sha256': file_hash(pointer), 'source_members': source['members'],
        'training_manifest_sha256': source['manifest_sha256'], 'code_revision': code_revision,
        'recipe': source['recipe'], 'mode': source['mode'], 'research_only': True,
        'independent_test_passed': False, 'background_class': False, 'localizes_defects': False,
        'parity': comparisons, 'status': 'validated_export',
        'limitations': ['classification_of_candidate_patches_only', 'provider_labels_not_expert_ground_truth',
                        'counts_depend_on_point_proposals', 'no_direction_subtypes_or_physical_conversion_claim']}
    write_json(output/'manifest.json', manifest)
    return manifest
