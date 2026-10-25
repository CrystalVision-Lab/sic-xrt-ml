"""Fine-label (class + direction subtype) patch classifiers.

Every architecture keeps the frozen input contract (uint8 RGB / 255, NCHW 128x128,
local contrast normalisation inside the model) and predicts the 10 provider fine
labels. Coarse BPD/TED/TSD probabilities are the sums of the fine probabilities of
each class group, so both outputs always agree on their totals.

Global pooling uses ``mean`` (deterministic CUDA backward), not AdaptiveAvgPool2d.
"""

import torch
from torch import nn

from ..data.provider_subtype import CLASSES, FINE_LABELS, FINE_TO_CLASS
from .model import PatchClassifier

SMALL = "xrt_local_contrast_subtype_cnn_s/v1"
MEDIUM = "xrt_local_contrast_subtype_cnn_m/v1"
MEDIUM_BN = "xrt_local_contrast_subtype_cnn_mb/v1"
MEDIUM_BN_CENTER = "xrt_local_contrast_subtype_cnn_mbc/v1"
ARCHITECTURES = (SMALL, MEDIUM, MEDIUM_BN, MEDIUM_BN_CENTER)


def _block(cin, cout, batch_norm=False):
    def norm():
        return nn.BatchNorm2d(cout) if batch_norm else nn.GroupNorm(8, cout)

    return [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        norm(),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        norm(),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ]


class SubtypeClassifier(nn.Module):
    """``s``: the v1 8/16/32 strided trunk with a 10-way head (same capacity as v1).

    ``m``: four 2-conv GroupNorm(8) stages (32/64/128/256 channels, 128 -> 8 px map),
    global mean pooling, dropout 0.2. ``mb``: the same trunk with BatchNorm.
    ``mbc``: ``mb`` pooled over the central 4x4 cells of the 8x8 map only (the
    labelled point is the patch centre; neighbours near the border are context).
    """

    def __init__(self, architecture=MEDIUM):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError("ARCHITECTURE")
        self.architecture = architecture
        if architecture == SMALL:
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.dropout = nn.Identity()
            width = 32
        else:
            layers = []
            for cin, cout in ((3, 32), (32, 64), (64, 128), (128, 256)):
                layers += _block(cin, cout, architecture != MEDIUM)
            self.features = nn.Sequential(*layers)
            self.dropout = nn.Dropout(0.2)
            width = 256
        self.head = nn.Linear(width, len(FINE_LABELS))

    def forward(self, x):
        f = self.features(PatchClassifier.contrast(x))
        if self.architecture == MEDIUM_BN_CENTER:
            f = f[:, :, 2:6, 2:6]
        return self.head(self.dropout(f.mean((2, 3))))


def group_matrix():
    """[10, 3] 0/1 matrix: fine probabilities @ G = coarse probabilities."""
    g = torch.zeros(len(FINE_LABELS), len(CLASSES))
    for i, c in enumerate(FINE_TO_CLASS):
        g[i, c] = 1.0
    return g


class SubtypeScoreExport(nn.Module):
    """(probabilities [N,3] = summed coarse, subtype_probabilities [N,10]).

    ``coarse_only`` returns just the summed [N,3] scores (single-output graph).
    """

    def __init__(self, model, coarse_only=False):
        super().__init__()
        self.model = model
        self.coarse_only = coarse_only
        self.register_buffer("groups", group_matrix())

    def forward(self, x):
        fine = torch.softmax(self.model(x), dim=1)
        if self.coarse_only:
            return fine @ self.groups
        return fine @ self.groups, fine
