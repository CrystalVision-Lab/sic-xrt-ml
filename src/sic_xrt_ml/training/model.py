"""A new compact baseline, not a reconstruction of the existing ONNX architecture."""

import torch
import torch.nn.functional as F
from torch import nn

ARCHITECTURE = "xrt_local_contrast_cnn/v1"


class PatchClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        if num_classes not in (3, 4):
            raise ValueError("NUM_CLASSES")
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # 3 = BPD/TED/TSD (v1 data); 4 adds NOT_DEFECT (v2 data). Same trunk.
        self.head = nn.Linear(32, num_classes)

    @staticmethod
    def contrast(x):
        mean = F.avg_pool2d(x, 9, stride=1, padding=4, count_include_pad=True)
        square = F.avg_pool2d(x * x, 9, stride=1, padding=4, count_include_pad=True)
        return (x - mean) / torch.sqrt(torch.clamp(square - mean * mean, min=0) + 1e-4)

    def forward(self, x):
        return self.head(self.features(self.contrast(x)).flatten(1))


class ScoreExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)
