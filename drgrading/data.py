"""Label loading, data partitioning and PyTorch datasets."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .config import (CV_FOLDS, DDR_TEST_FRACTION, DDR_UNGRADABLE_LABEL,
                     DDR_VAL_FRACTION, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                     TEST_FRACTION, VAL_FRACTION)
from .preprocessing import load_rgb, preprocess_image


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def load_aptos_labels(csv_path):
    """Read the APTOS 2019 train.csv file (columns id_code, diagnosis).

    Returns a DataFrame with columns ``image`` (file stem) and ``label`` (0-4).
    """
    df = pd.read_csv(csv_path, dtype={"id_code": str})
    return pd.DataFrame({"image": df["id_code"].astype(str),
                         "label": df["diagnosis"].astype(int)})


def load_ddr_labels(ddr_root):
    """Read the DDR grading label files and drop ungradable images.

    The DDR release provides ``train.txt``, ``valid.txt`` and ``test.txt`` in the
    DR_grading folder, each line being ``<file name> <grade>``. The three files
    are merged because a new stratified 70/15/15 split is drawn afterwards.
    """
    ddr_root = Path(ddr_root)
    rows = []
    for name in ("train.txt", "valid.txt", "test.txt"):
        label_file = ddr_root / name
        if not label_file.exists():
            continue
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                rows.append((Path(parts[0]).stem, int(parts[1])))
    if not rows:
        raise FileNotFoundError(f"No DDR label file found in {ddr_root}")

    df = pd.DataFrame(rows, columns=["image", "label"]).drop_duplicates("image")
    return df[df["label"] != DDR_UNGRADABLE_LABEL].reset_index(drop=True)


def to_task_labels(labels, task):
    """Map 0-4 severity grades to the labels of the requested task.

    binary     : 0 = No DR, 1 = DR (grades 1 to 4)
    multiclass : grades kept unchanged
    """
    labels = np.asarray(labels, dtype=int)
    if task == "binary":
        return (labels > 0).astype(int)
    return labels


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def make_holdout_split(df, test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION, seed=42):
    """Stratified train / validation / test split used for APTOS 2019.

    The test set (``test_fraction`` of all images) is drawn first and never used
    for model selection. The remaining development partition is then divided
    into training and validation (``val_fraction`` of the development set).
    Stratification uses the five severity grades, so the same split serves both
    the binary and the five-class task.
    """
    dev, test = train_test_split(df, test_size=test_fraction, stratify=df["label"],
                                 random_state=seed)
    train, val = train_test_split(dev, test_size=val_fraction, stratify=dev["label"],
                                  random_state=seed)
    return _assemble(train, val, test)


def make_three_way_split(df, val_fraction=DDR_VAL_FRACTION, test_fraction=DDR_TEST_FRACTION, seed=42):
    """Stratified train / validation / test split given as fractions of the whole set (DDR)."""
    dev, test = train_test_split(df, test_size=test_fraction, stratify=df["label"],
                                 random_state=seed)
    relative_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(dev, test_size=relative_val, stratify=dev["label"],
                                  random_state=seed)
    return _assemble(train, val, test)


def _assemble(train, val, test):
    parts = []
    for name, part in (("train", train), ("val", val), ("test", test)):
        part = part.copy()
        part["split"] = name
        parts.append(part)
    return pd.concat(parts).sort_values(["split", "image"]).reset_index(drop=True)


def make_cv_folds(dev_df, seeds, n_splits=CV_FOLDS):
    """Stratified k-fold assignments of the development partition, repeated for each seed.

    Returns a DataFrame with columns image, label, seed, fold where ``fold`` is
    the index of the fold in which the image is used for validation.
    """
    records = []
    labels = dev_df["label"].to_numpy()
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            for i in val_idx:
                records.append((dev_df.iloc[i]["image"], int(labels[i]), seed, fold))
    return pd.DataFrame(records, columns=["image", "label", "seed", "fold"])


def load_split(split_csv, split=None):
    """Read a split file written by make_splits.py, optionally keeping one partition."""
    df = pd.read_csv(split_csv, dtype={"image": str})
    if split is not None:
        names = [split] if isinstance(split, str) else list(split)
        df = df[df["split"].isin(names)]
    return df.reset_index(drop=True)


def describe_split(df, task="multiclass"):
    """Class counts per partition (used for logging)."""
    tmp = df.copy()
    tmp["target"] = to_task_labels(tmp["label"], task)
    return tmp.pivot_table(index="target", columns="split", values="image",
                           aggfunc="count", fill_value=0)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def build_transform():
    """Tensor conversion followed by fixed ImageNet normalisation (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def resolve_image_path(image_dir, stem, extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    """Find the file corresponding to an image stem inside a folder."""
    image_dir = Path(image_dir)
    for ext in extensions:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image named {stem} in {image_dir}")


class FundusDataset(Dataset):
    """Fundus images with their task labels.

    By default the images are expected to be already preprocessed by
    preprocess.py. If ``preprocess_kwargs`` is given, raw images are
    preprocessed on the fly with those arguments instead.
    """

    def __init__(self, df, image_dir, task, preprocess_kwargs=None):
        self.image_dir = Path(image_dir)
        self.images = df["image"].astype(str).tolist()
        self.labels = to_task_labels(df["label"].to_numpy(), task)
        self.task = task
        self.preprocess_kwargs = preprocess_kwargs
        self.transform = build_transform()

    def __len__(self):
        return len(self.images)

    def load_image(self, index):
        """Return the preprocessed RGB uint8 array of an item (also used for display)."""
        image = load_rgb(resolve_image_path(self.image_dir, self.images[index]))
        if self.preprocess_kwargs is not None:
            image = preprocess_image(image, **self.preprocess_kwargs)
        elif image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        return image

    def __getitem__(self, index):
        x = self.transform(self.load_image(index))
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


class FeatureDataset(Dataset):
    """Pre-extracted backbone features (frozen regime).

    ``features`` maps a branch name ("vit", "resnet") to a tensor of shape (N, D).
    """

    def __init__(self, features, labels, indices=None):
        self.features = features
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        n = len(self.labels)
        self.indices = np.arange(n) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return {k: v[i] for k, v in self.features.items()}, self.labels[i]
