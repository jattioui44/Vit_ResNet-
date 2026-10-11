"""Hybrid ViT-Base/16 + ResNet50 architecture.

Overview
--------
    image (224x224x3)
      |-- ViT-Base/16 (frozen) -> [CLS] token (768) -> FC 768->512 + ReLU  -> f_vit
      |-- ResNet50    (frozen) -> global avg pool (2048) -> FC 2048->512 + ReLU -> f_resnet
    f = concat(f_vit, f_resnet)                       (1024)
    head: 1024 -> 512 -> 256 -> 128 -> 64 -> k        (ReLU, dropout 0.3 after the
                                                       first three hidden layers)
    k = 1 (sigmoid, binary) or k = 5 (softmax, five-class)

The model is split in two modules:

* ``BackboneExtractor`` holds the pretrained backbones. In the frozen regime
  its outputs never change, so they are computed once and cached.
* ``FusionClassifier`` holds the trainable part (projections, fusion, head).

``HybridModel`` chains both and is used end to end for fine-tuning and for the
explainability methods, which need gradients with respect to the input pixels.
"""

import timm
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from .config import (DROPOUT, HEAD_DIMS, HEAD_DROPOUT_LAYERS, NUM_OUTPUTS,
                     PROJ_DIM, RESNET_DIM, VIT_DIM, VIT_MODEL_NAME)

BRANCH_DIMS = {"vit": VIT_DIM, "resnet": RESNET_DIM}
REGIMES = ("frozen", "partial", "full")
FUSIONS = ("concat", "weighted_sum", "product", "bilinear")


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

def build_vit(pretrained=True):
    """ViT-Base/16 without classification head; returns the 768-d [CLS] embedding."""
    return timm.create_model(VIT_MODEL_NAME, pretrained=pretrained, num_classes=0)


def build_resnet50(pretrained=True):
    """ResNet50 without its final FC layer; returns the 2048-d pooled embedding."""
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    return model


class BackboneExtractor(nn.Module):
    """Runs the selected pretrained backbones and returns their embeddings."""

    def __init__(self, branches=("vit", "resnet"), pretrained=True, regime="frozen"):
        super().__init__()
        if not branches:
            raise ValueError("At least one branch is required")
        self.branches = tuple(branches)
        self.vit = build_vit(pretrained) if "vit" in self.branches else None
        self.resnet = build_resnet50(pretrained) if "resnet" in self.branches else None
        self.regime = None
        self.set_regime(regime)

    def set_regime(self, regime):
        """Select which backbone parameters receive gradient updates.

        frozen  : no backbone parameter is trained (proposed configuration)
        partial : last two encoder blocks of ViT (and its final norm) and last two
                  bottleneck blocks of ResNet50 are trained
        full    : every backbone parameter is trained
        """
        if regime not in REGIMES:
            raise ValueError(f"Unknown regime {regime}, expected one of {REGIMES}")
        self.regime = regime
        for p in self.parameters():
            p.requires_grad = regime == "full"

        if regime == "partial":
            trainable = []
            if self.vit is not None:
                trainable += [*self.vit.blocks[-2:], self.vit.norm]
            if self.resnet is not None:
                trainable += [*self.resnet.layer4[-2:]]
            for module in trainable:
                for p in module.parameters():
                    p.requires_grad = True
        self.train(self.training)

    def train(self, mode=True):
        """Keep frozen backbones in evaluation mode.

        Batch-normalisation statistics of ResNet50 are never updated: the
        dataset is small and batches contain only 16 to 32 images.
        """
        super().train(mode)
        if self.regime == "frozen":
            super().train(False)
        else:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x):
        features = {}
        if self.vit is not None:
            features["vit"] = self.vit(x)
        if self.resnet is not None:
            features["resnet"] = self.resnet(x)
        return features


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    """Concatenation of the projected embeddings (proposed, no extra parameter)."""

    def __init__(self, dim, n_inputs):
        super().__init__()
        self.out_dim = dim * n_inputs

    def forward(self, embeddings):
        return torch.cat(embeddings, dim=1)


class WeightedSumFusion(nn.Module):
    """Learnable scalar weight per branch (2 extra parameters)."""

    def __init__(self, dim, n_inputs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs))
        self.out_dim = dim

    def forward(self, embeddings):
        return sum(w * e for w, e in zip(self.weights, embeddings))


class ProductFusion(nn.Module):
    """Element-wise product of the embeddings (no extra parameter)."""

    def __init__(self, dim, n_inputs):
        super().__init__()
        self.out_dim = dim

    def forward(self, embeddings):
        out = embeddings[0]
        for e in embeddings[1:]:
            out = out * e
        return out


class LowRankBilinearFusion(nn.Module):
    """Low-rank bilinear pooling: (U a) * (V b).

    With dim = 512 and rank = 256 the two projections add 2 x 512 x 256
    = 262,144 parameters.
    """

    def __init__(self, dim, n_inputs, rank=256):
        super().__init__()
        if n_inputs != 2:
            raise ValueError("Bilinear fusion requires exactly two branches")
        self.u = nn.Linear(dim, rank, bias=False)
        self.v = nn.Linear(dim, rank, bias=False)
        self.out_dim = rank

    def forward(self, embeddings):
        a, b = embeddings
        return self.u(a) * self.v(b)


def build_fusion(name, dim, n_inputs):
    if n_inputs == 1:
        return ConcatFusion(dim, 1)  # identity for a single branch
    if name == "concat":
        return ConcatFusion(dim, n_inputs)
    if name == "weighted_sum":
        return WeightedSumFusion(dim, n_inputs)
    if name == "product":
        return ProductFusion(dim, n_inputs)
    if name == "bilinear":
        return LowRankBilinearFusion(dim, n_inputs)
    raise ValueError(f"Unknown fusion {name}, expected one of {FUSIONS}")


# ---------------------------------------------------------------------------
# Trainable part
# ---------------------------------------------------------------------------

def build_head(in_dim, num_outputs, hidden_dims=HEAD_DIMS, dropout=DROPOUT,
               dropout_layers=HEAD_DROPOUT_LAYERS):
    """Fully connected head: in_dim -> 512 -> 256 -> 128 -> 64 -> num_outputs."""
    layers = []
    dims = [in_dim, *hidden_dims]
    for i in range(len(hidden_dims)):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        if i < dropout_layers:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], num_outputs))
    return nn.Sequential(*layers)


class FusionClassifier(nn.Module):
    """Projection layers, fusion and classification head (all trainable)."""

    def __init__(self, task, branches=("vit", "resnet"), fusion="concat",
                 proj_dim=PROJ_DIM, dropout=DROPOUT):
        super().__init__()
        self.task = task
        self.branches = tuple(branches)
        self.projections = nn.ModuleDict({
            b: nn.Sequential(nn.Linear(BRANCH_DIMS[b], proj_dim), nn.ReLU(inplace=True))
            for b in self.branches
        })
        self.fusion = build_fusion(fusion, proj_dim, len(self.branches))
        self.head = build_head(self.fusion.out_dim, NUM_OUTPUTS[task], dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Glorot uniform initialisation of every linear layer, zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features):
        embeddings = [self.projections[b](features[b]) for b in self.branches]
        return self.head(self.fusion(embeddings))


class HybridModel(nn.Module):
    """End-to-end model: image -> logits."""

    def __init__(self, extractor, classifier):
        super().__init__()
        self.extractor = extractor
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.extractor(x))


def build_model(task, branches=("vit", "resnet"), fusion="concat", regime="frozen",
                pretrained=True, dropout=DROPOUT):
    """Create the extractor and the classifier for a given configuration."""
    extractor = BackboneExtractor(branches, pretrained=pretrained, regime=regime)
    classifier = FusionClassifier(task, branches, fusion, dropout=dropout)
    return HybridModel(extractor, classifier)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, config):
    """Save the trained weights and the configuration needed to rebuild the model.

    In the frozen regime only the classifier is stored: the backbones are the
    public pretrained weights and are reloaded from timm / torchvision.
    """
    state = {
        "config": dict(config),
        "classifier": model.classifier.state_dict(),
        "extractor": None if config["regime"] == "frozen" else model.extractor.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, device="cpu", pretrained_backbones=True):
    """Rebuild a HybridModel from a checkpoint written by save_checkpoint."""
    state = torch.load(path, map_location="cpu")
    cfg = state["config"]
    # When the full extractor is stored, downloading the pretrained weights is unnecessary.
    pretrained = pretrained_backbones and state["extractor"] is None
    model = build_model(cfg["task"], cfg["branches"], cfg["fusion"], cfg["regime"],
                        pretrained=pretrained)
    model.classifier.load_state_dict(state["classifier"])
    if state["extractor"] is not None:
        model.extractor.load_state_dict(state["extractor"])
    model.to(device).eval()
    return model, cfg


class TwoLogitWrapper(nn.Module):
    """Expose a single-logit binary model as a two-output model.

    Output 0 is -z (evidence for No DR) and output 1 is +z (evidence for DR), so
    the attribution methods can target either class with the same code path
    used for the five-class model. Multiclass models are returned unchanged.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z = self.model(x)
        return torch.cat([-z, z], dim=1)


def as_attribution_model(model, task):
    return TwoLogitWrapper(model) if task == "binary" else model
