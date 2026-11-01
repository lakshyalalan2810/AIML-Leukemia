"""Unified Grad-CAM explainability interface for ConvNeXtV2-Tiny and EfficientNet-B1."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Computes Grad-CAM heatmaps for any PyTorch CNN model with specified target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """Compute 2D normalized Grad-CAM heatmap for target_class."""
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM currently requires a single NCHW input tensor")
        self.activations = None
        self.gradients = None
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        score = logits[:, target_class].sum()

        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients")
        if self.activations.ndim != 4 or self.gradients.ndim != 4:
            raise RuntimeError(
                "Grad-CAM target layer must produce 4D NCHW activations and gradients; "
                f"got activations={tuple(self.activations.shape)}, gradients={tuple(self.gradients.shape)}"
            )
        if self.activations.shape != self.gradients.shape:
            raise RuntimeError("Grad-CAM activation and gradient tensor shapes do not match")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to focus on positive attribution

        cam_np = cam.squeeze().cpu().numpy()
        min_val, max_val = cam_np.min(), cam_np.max()
        if max_val - min_val > 1e-8:
            cam_norm = (cam_np - min_val) / (max_val - min_val)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm


def generate_gradcam_overlay(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay normalized 2D heatmap on the original PIL image."""
    import matplotlib.cm as cm

    w, h = original_img.size
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)
    colored_heatmap_img = Image.fromarray(colored_heatmap_uint8)

    blended = Image.blend(original_img.convert("RGB"), colored_heatmap_img, alpha=alpha)
    return blended


def compute_border_attention_metrics(heatmap: np.ndarray, border_fraction: float = 0.10) -> dict[str, float]:
    """Measure normalized Grad-CAM energy in the outer border and its complement."""
    heatmap = np.asarray(heatmap, dtype=float)
    if heatmap.ndim != 2 or heatmap.size == 0 or not np.all(np.isfinite(heatmap)):
        raise ValueError("Grad-CAM heatmap must be a non-empty finite 2D array")
    if np.any(heatmap < 0) or not 0 < border_fraction < 0.5:
        raise ValueError("Grad-CAM heatmap must be non-negative and border_fraction must be in (0, 0.5)")

    height, width = heatmap.shape
    border_h = max(1, int(np.ceil(height * border_fraction)))
    border_w = max(1, int(np.ceil(width * border_fraction)))
    border = np.zeros_like(heatmap, dtype=bool)
    border[:border_h, :] = border[-border_h:, :] = True
    border[:, :border_w] = border[:, -border_w:] = True
    center = ~border

    total = float(heatmap.sum())
    if total == 0:
        return {"border_energy_fraction": 0.0, "center_energy_fraction": 0.0, "center_to_border_ratio": 0.0}
    border_energy = float(heatmap[border].sum())
    center_energy = float(heatmap[center].sum())
    ratio = float(heatmap[center].mean() / (heatmap[border].mean() + 1e-12)) if center.any() else 0.0
    return {
        "border_energy_fraction": border_energy / total,
        "center_energy_fraction": center_energy / total,
        "center_to_border_ratio": ratio,
    }
