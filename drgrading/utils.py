"""Small helpers used across the package."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """Fix every random number generator used by the pipeline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels: slightly slower but repeatable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name=None):
    """Return the requested device, or CUDA when available, else CPU."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    """Create a directory (and its parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_serialisable(obj):
    """Convert numpy and torch scalars or arrays into plain Python objects."""
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path):
    """Write an object to a JSON file with indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serialisable(obj), f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(module, trainable_only=True):
    """Number of (trainable) parameters of a module."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
