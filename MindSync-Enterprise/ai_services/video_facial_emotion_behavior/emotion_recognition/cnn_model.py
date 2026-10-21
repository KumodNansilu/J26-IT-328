"""Custom FER2013 CNN (PyTorch): 48x48 gray -> 7 logits in EMOTIONS order.

Architecture (approved): 3 conv blocks (32/64/128) + BN + ReLU + pool +
dropout, FC 256 -> 7. ~600k params, CPU-friendly.
preprocess_crop() is the SINGLE conversion from Step-3 BGR FaceFrame.crop
to model input — training and inference share it so they cannot drift.
"""
from __future__ import annotations

MODEL_INPUT_SIZE = 48
MODEL_INPUT_SHAPE = (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover — torch required for train/infer
    torch = None  # type: ignore
    nn = None  # type: ignore


def preprocess_crop(bgr_crop):
    """BGR face crop -> (1,48,48) float32 tensor in 0..1. Raises on invalid."""
    import cv2
    import numpy as np

    if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
        raise ValueError("Empty face crop — cannot run emotion inference.")
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    if torch is None:
        return arr.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    return torch.from_numpy(arr).unsqueeze(0)  # (1,48,48)


if nn is not None:
    class FerCNN(nn.Module):
        """Small custom CNN: (N,1,48,48) -> (N,7) logits, EMOTIONS order."""

        def __init__(self, num_classes: int = 7, dropout: float = 0.25,
                     fc_dropout: float = 0.5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(dropout),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(dropout),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 6 * 6, 256), nn.ReLU(inplace=True),
                nn.Dropout(fc_dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    def build_model(num_classes: int = 7) -> "FerCNN":
        return FerCNN(num_classes=num_classes)
else:  # pragma: no cover
    FerCNN = None  # type: ignore

    def build_model(num_classes: int = 7):
        raise RuntimeError("PyTorch (torch) is required but not installed.")

