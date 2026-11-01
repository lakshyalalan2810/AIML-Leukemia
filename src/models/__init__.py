"""Model definitions for ConvNeXtV2-Tiny and EfficientNet-B1."""

from .convnextv2_tiny import ConvNeXtV2TinyClassifier, create_convnextv2_tiny
from .efficientnet_b1 import EfficientNetB1Classifier, create_efficientnet_b1

__all__ = [
    "ConvNeXtV2TinyClassifier",
    "create_convnextv2_tiny",
    "EfficientNetB1Classifier",
    "create_efficientnet_b1",
]
