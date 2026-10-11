"""Gradient-based explanation methods.

All four methods explain the full image -> logit mapping of the hybrid model:

* Grad-CAM            : class-score gradients on the last convolutional block of
                        ResNet50, weighted sum of its feature maps, ReLU,
                        upsampled to 224 x 224 (pytorch-grad-cam)
* Integrated Gradients: path integral of the gradients from a black-image
                        baseline to the input, 50 steps (captum)
* Saliency            : absolute gradient of the class score w.r.t. the input (captum)
* SmoothGrad          : saliency averaged over n = 50 noisy copies of the input,
                        sigma = 0.15 x (max(x) - min(x)) (captum NoiseTunnel)

Pixel-level attributions (IG, Saliency, SmoothGrad) have one value per colour
channel; they are reduced to one map by taking the maximum absolute value over
the channels.
"""

import numpy as np
import torch
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import IMAGENET_MEAN, IMAGENET_STD
from .model import as_attribution_model

METHODS = ("Grad-CAM", "Integrated Gradients", "Saliency", "SmoothGrad")


def normalized_black_image(x):
    """Black image (all pixels 0) expressed in the normalised input space."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((torch.zeros_like(x) - mean) / std).expand_as(x).contiguous()


def denormalize(x):
    """Normalised tensor (1, 3, H, W) -> RGB array (H, W, 3) in [0, 1] for display."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    image = (x[0] * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).detach().cpu().numpy()


def to_heatmap(attribution, clip_percentile=99.0):
    """Reduce a (1, 3, H, W) attribution to an (H, W) map scaled to [0, 1].

    Extreme values are clipped at a high percentile so that a handful of
    pixels do not compress the rest of the map towards zero.
    """
    amap = attribution.detach().abs().amax(dim=1)[0].cpu().numpy()
    upper = np.percentile(amap, clip_percentile) if clip_percentile else amap.max()
    amap = np.clip(amap, 0, upper)
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-12)


class Explainer:
    """Compute the four attribution maps for one model."""

    def __init__(self, model, task, smoothgrad_samples=50, smoothgrad_noise=0.15,
                 ig_steps=50, internal_batch_size=10):
        self.task = task
        self.model = as_attribution_model(model, task).eval()
        self.smoothgrad_samples = smoothgrad_samples
        self.smoothgrad_noise = smoothgrad_noise
        self.ig_steps = ig_steps
        self.internal_batch_size = internal_batch_size

        self.saliency = Saliency(self.model)
        self.smoothgrad = NoiseTunnel(Saliency(self.model))
        self.integrated_gradients = IntegratedGradients(self.model)

        # Grad-CAM needs a convolutional layer: the last bottleneck of ResNet50.
        resnet = model.extractor.resnet
        self.gradcam = GradCAM(model=self.model, target_layers=[resnet.layer4[-1]]) \
            if resnet is not None else None

    def gradcam_map(self, x, target):
        if self.gradcam is None:
            raise RuntimeError("Grad-CAM requires the ResNet50 branch")
        # The backbones are frozen (requires_grad=False), so the ResNet50
        # activations only carry gradients if the input itself requires them.
        x = x.clone().requires_grad_(True)
        cam = self.gradcam(input_tensor=x, targets=[ClassifierOutputTarget(int(target))])
        return cam[0]

    def saliency_map(self, x, target):
        x = x.clone().requires_grad_(True)
        return to_heatmap(self.saliency.attribute(x, target=int(target), abs=True))

    def smoothgrad_attribution(self, x, target):
        """Raw SmoothGrad attribution (1, 3, H, W); also used by the faithfulness test."""
        sigma = self.smoothgrad_noise * (x.max() - x.min()).item()
        x = x.clone().requires_grad_(True)
        return self.smoothgrad.attribute(
            x, nt_type="smoothgrad", nt_samples=self.smoothgrad_samples,
            nt_samples_batch_size=self.internal_batch_size, stdevs=sigma,
            target=int(target), abs=True)

    def smoothgrad_map(self, x, target):
        return to_heatmap(self.smoothgrad_attribution(x, target))

    def integrated_gradients_map(self, x, target):
        attribution = self.integrated_gradients.attribute(
            x, baselines=normalized_black_image(x), target=int(target),
            n_steps=self.ig_steps, internal_batch_size=self.internal_batch_size)
        return to_heatmap(attribution)

    def explain(self, x, target, methods=METHODS):
        """Return a dict method name -> (H, W) heatmap in [0, 1]."""
        functions = {
            "Grad-CAM": self.gradcam_map,
            "Integrated Gradients": self.integrated_gradients_map,
            "Saliency": self.saliency_map,
            "SmoothGrad": self.smoothgrad_map,
        }
        return {m: functions[m](x, target) for m in methods}
