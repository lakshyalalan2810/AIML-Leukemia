"""ConvNeXtV2-Tiny classifier for leukemia single-cell microscopy."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

CONVNEXT_RECIPE = "convnextv2_tiny.fcmae_ft_in1k"
EXPECTED_PARAMS = 27868034
FEATURE_DIM = 768


class ConvNeXtV2TinyClassifier(nn.Module):
    """ConvNeXtV2-Tiny model with GAP, LayerNorm(768), and Linear(768, 2)."""

    def __init__(self, pretrained: bool = True, num_classes: int = 2) -> None:
        super().__init__()
        # Load backbone from timm with num_classes=0 to get pooled features or feature map
        self.backbone = timm.create_model(
            CONVNEXT_RECIPE,
            pretrained=pretrained,
            num_classes=0,  # removes head, keeps GAP and LayerNorm(768)
        )
        # timm's convnextv2 with num_classes=0 has head.global_pool (SelectAdaptivePool2d) and head.norm (LayerNorm2d/LayerNorm)
        # Check feature dimension
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # Shape: (B, 768)
        logits = self.classifier(features)  # Shape: (B, 2)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the appropriate final spatial block for Grad-CAM."""
        # Backbone stages[-1].blocks[-1]
        return self.backbone.stages[-1].blocks[-1]

    def classifier_named_parameters(self):
        return ((f"classifier.{name}", param) for name, param in self.classifier.named_parameters())

    def set_warmup_mode(self) -> None:
        self.eval()
        self.training = True
        self.classifier.train()


def create_convnextv2_tiny(pretrained: bool = True) -> ConvNeXtV2TinyClassifier:
    """Instantiate and verify ConvNeXtV2-Tiny classifier."""
    model = ConvNeXtV2TinyClassifier(pretrained=pretrained, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    if total_params != EXPECTED_PARAMS:
        raise ValueError(
            f"Expected {EXPECTED_PARAMS} parameters for ConvNeXtV2-Tiny, got {total_params}"
        )
    return model
