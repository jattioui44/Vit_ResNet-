"""Loss functions and class-imbalance strategies.

The proposed configuration uses plain (binary) cross-entropy. The other
strategies reproduce the class-imbalance comparison:

* weighted_ce : cross-entropy with class weights w_c = N / n_c
* focal       : focal loss with gamma = 2 and no class weights
* oversampling: random duplication of minority-class training samples until
                every class matches the majority class count
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FOCAL_GAMMA, NUM_OUTPUTS

LOSSES = ("ce", "weighted_ce", "focal")


def class_counts(labels, num_classes):
    return np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)


def inverse_frequency_weights(labels, num_classes):
    """w_c = N / n_c, where N is the training-set size and n_c the class count."""
    counts = class_counts(labels, num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / counts, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss, FL = (1 - p_t)^gamma * CE, for binary or multiclass logits."""

    def __init__(self, gamma=FOCAL_GAMMA, binary=False):
        super().__init__()
        self.gamma = gamma
        self.binary = binary

    def forward(self, logits, targets):
        if self.binary:
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        else:
            ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


class BinaryLoss(nn.Module):
    """Wrap a BCE-style criterion so that it accepts integer labels of shape (N,)."""

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, targets):
        return self.criterion(logits.view(-1), targets.float().view(-1))


def build_criterion(task, strategy, train_labels, gamma=FOCAL_GAMMA):
    """Create the training criterion for a task and an imbalance strategy."""
    if strategy not in LOSSES:
        raise ValueError(f"Unknown loss {strategy}, expected one of {LOSSES}")

    if task == "binary":
        if strategy == "ce":
            return BinaryLoss(nn.BCEWithLogitsLoss())
        if strategy == "weighted_ce":
            # For a single logit, w_1 / w_0 = n_0 / n_1 is applied to positives.
            counts = class_counts(train_labels, 2).astype(np.float64)
            pos_weight = torch.tensor([counts[0] / max(counts[1], 1.0)], dtype=torch.float32)
            return BinaryLoss(nn.BCEWithLogitsLoss(pos_weight=pos_weight))
        return BinaryLoss(FocalLoss(gamma, binary=True))

    if strategy == "ce":
        return nn.CrossEntropyLoss()
    if strategy == "weighted_ce":
        return nn.CrossEntropyLoss(weight=inverse_frequency_weights(train_labels, NUM_OUTPUTS[task]))
    return FocalLoss(gamma, binary=False)


def oversample_indices(labels, seed=42):
    """Indices of a training set where every class is duplicated up to the majority count."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    target = np.bincount(labels).max()
    indices = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        extra = rng.choice(idx, size=target - len(idx), replace=True) if len(idx) < target else []
        indices.append(np.concatenate([idx, np.asarray(extra, dtype=int)]))
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return indices
