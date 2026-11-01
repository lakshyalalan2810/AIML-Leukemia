"""EfficientNet-B1 classifier for leukemia single-cell microscopy."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

EFFICIENTNET_RECIPE = "tf_efficientnet_b1.aa_in1k"
EXPECTED_PARAMS = 6515746
FEATURE_DIM = 1280
DEFAULT_DROPOUT = 0.2


class EfficientNetB1Classifier(nn.Module):
    """EfficientNet-B1 model with GAP, Dropout(0.2), and Linear(1280, 2)."""

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        drop_rate: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            EFFICIENTNET_RECIPE,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the target convolutional head for Grad-CAM."""
        return self.model.conv_head

    def classifier_named_parameters(self):
        return ((f"model.classifier.{name}", param) for name, param in self.model.classifier.named_parameters())

    def set_warmup_mode(self) -> None:
        self.eval()
        self.training = True
        self.model.training = True  # keeps timm's functional classifier dropout active
        self.model.classifier.train()


def create_efficientnet_b1(pretrained: bool = True, drop_rate: float = DEFAULT_DROPOUT) -> EfficientNetB1Classifier:
    """Instantiate and verify EfficientNet-B1 classifier."""
    model = EfficientNetB1Classifier(pretrained=pretrained, num_classes=2, drop_rate=drop_rate)
    total_params = sum(p.numel() for p in model.parameters())
    if total_params != EXPECTED_PARAMS:
        raise ValueError(
            f"Expected {EXPECTED_PARAMS} parameters for EfficientNet-B1, got {total_params}"
        )
    return model
