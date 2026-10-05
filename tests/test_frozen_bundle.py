import torch

from sic_xrt_ml.export.frozen_bundle import ExportProbabilities
from sic_xrt_ml.training.bpd_context import EqualProbabilityEnsemble
from sic_xrt_ml.training.orientation_stability import OrientationMean
from sic_xrt_ml.training.spatial_robustness import SpatialClassifier


def test_export_rotations_preserve_original_c4_probabilities():
    torch.manual_seed(7)
    original = OrientationMean(EqualProbabilityEnsemble([SpatialClassifier(64), SpatialClassifier(64)]), 'c4').eval()
    exported = ExportProbabilities(original).eval()
    probe = torch.rand(2, 3, 128, 128)
    with torch.inference_mode():
        assert torch.allclose(original(probe).softmax(1), exported(probe), atol=1e-6)
