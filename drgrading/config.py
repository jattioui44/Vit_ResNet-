"""Global constants shared by all scripts.

Every value below corresponds to a setting described in the paper. The
command-line scripts expose most of them as arguments so that ablation
experiments can override them without editing this file.
"""

# Reproducibility
SEED = 42

# Input resolution expected by both backbones
IMAGE_SIZE = 224

# Preprocessing
# Grey level below which a pixel is considered part of the black border.
BORDER_THRESHOLD = 10
# Contrast scaling factor of the mean-anchored transform, selected by grid
# search over {1.5, 2.0, 2.5, 3.0} on the validation set.
CONTRAST_ALPHA = 2.0

# Normalisation applied after preprocessing. Fixed ImageNet statistics are used
# so that no statistic is estimated on the fundus data (no leakage).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Backbones
VIT_MODEL_NAME = "vit_base_patch16_224_in21k"  # timm, ImageNet-21k weights
VIT_DIM = 768        # dimension of the [CLS] token output
RESNET_DIM = 2048    # dimension of the ResNet50 global average pooling output
PROJ_DIM = 512       # dimension of each projected embedding

# Classification head: 1024 -> 512 -> 256 -> 128 -> 64 -> k
HEAD_DIMS = (512, 256, 128, 64)
# Dropout is applied after the first three hidden layers only.
HEAD_DROPOUT_LAYERS = 3
DROPOUT = 0.3

# Optimisation (fixed a priori, not tuned)
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
BATCH_SIZE = {"binary": 32, "multiclass": 16}
MAX_EPOCHS = 100
PATIENCE = 10
FEATURE_BATCH_SIZE = 64
FOCAL_GAMMA = 2.0

# Data partitioning (APTOS 2019)
TEST_FRACTION = 0.15   # held-out test set
VAL_FRACTION = 0.20    # validation share of the 85% development partition

# Data partitioning (DDR, 70/15/15)
DDR_VAL_FRACTION = 0.15
DDR_TEST_FRACTION = 0.15
DDR_UNGRADABLE_LABEL = 5

# Cross-validation
CV_FOLDS = 5
CV_SEEDS = (42, 123, 456)

# Tasks
TASKS = ("binary", "multiclass")
NUM_OUTPUTS = {"binary": 1, "multiclass": 5}
CLASS_NAMES = {
    "binary": ["No DR", "DR"],
    "multiclass": ["No DR", "Mild", "Moderate", "Severe", "PDR"],
}

# Binary decision threshold (fixed a priori, not tuned)
BINARY_THRESHOLD = 0.5

# Population DR prevalence used to report PPV and NPV
DR_PREVALENCE = 0.34
