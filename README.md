# Hybrid ViT-ResNet50 for Diabetic Retinopathy Detection and Grading

Code accompanying the paper *"A Hybrid Vision Transformer-ResNet50 Framework for
Automated Detection and Severity Grading of Diabetic Retinopathy from Fundus Images"*.

The pipeline has three stages:

1. **Preprocessing**: black-border removal (largest-contour crop) and
   mean-anchored contrast enhancement, `I' = mu + alpha (I - mu)` with `alpha = 2`,
   followed by resizing to 224 x 224.
2. **Hybrid feature extraction**: two frozen ImageNet-pretrained backbones run in
   parallel. ViT-Base/16 provides the 768-d `[CLS]` embedding (global context) and
   ResNet50 the 2048-d pooled embedding (local texture). Each is projected to
   512 dimensions and the two vectors are concatenated (1024-d).
3. **Classification**: a fully connected head
   `1024 -> 512 -> 256 -> 128 -> 64 -> k` (ReLU, dropout 0.3), with `k = 1`
   (sigmoid) for binary DR detection and `k = 5` (softmax) for severity grading.

Four gradient-based explanation methods are provided (Grad-CAM, Integrated
Gradients, Saliency, SmoothGrad), together with a pixel-masking faithfulness test.

## Repository structure

```
.
|-- drgrading/                 Python package
|   |-- config.py              all hyperparameters and constants
|   |-- preprocessing.py       border removal, contrast enhancement, resizing
|   |-- data.py                label loading, stratified splits, datasets
|   |-- model.py               backbones, fusion strategies, head, checkpoints
|   |-- losses.py              cross-entropy, weighted CE, focal loss, oversampling
|   |-- engine.py              feature extraction, training loop, prediction
|   |-- metrics.py             metrics, confidence intervals, McNemar, Holm
|   |-- explainability.py      Grad-CAM, Integrated Gradients, Saliency, SmoothGrad
|   `-- plots.py               confusion matrices, curves, attribution overlays
|-- preprocess.py              preprocess a folder of fundus images
|-- make_splits.py             train / validation / test partition
|-- train.py                   training and evaluation (binary or five-class)
|-- cross_validate.py          repeated stratified 5-fold cross-validation
|-- evaluate.py                evaluate a checkpoint or a predictions file
|-- compare_models.py          McNemar tests with Holm-Bonferroni correction
|-- explain.py                 attribution maps for selected images
|-- faithfulness.py            pixel-masking perturbation test
|-- splits/                    split and fold files
`-- requirements.txt
```

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if needed
(see https://pytorch.org). The pretrained weights are downloaded automatically
the first time a model is built: ViT-Base/16 ImageNet-21k from `timm`
(`vit_base_patch16_224_in21k`) and ResNet50 `IMAGENET1K_V1` from `torchvision`.

## Data

The images are not redistributed. Download them from the original sources:

* **APTOS 2019 Blindness Detection**: https://www.kaggle.com/competitions/aptos2019-blindness-detection
  (use `train.csv` and `train_images/`).
* **DDR**: https://github.com/nkicsl/DDR-dataset (folder `DR_grading`).

Expected layout (any other location can be passed on the command line):

```
data/
|-- aptos2019/
|   |-- train.csv
|   `-- train_images/*.png
`-- DDR/
    `-- DR_grading/
        |-- train.txt  valid.txt  test.txt
        `-- images/*.jpg
```

## Usage

### 1. Preprocessing

```bash
python preprocess.py --input-dir data/aptos2019/train_images \
                     --output-dir data/aptos2019/preprocessed
```

DDR images have no black border, so border removal is skipped:

```bash
python preprocess.py --input-dir data/DDR/DR_grading/images \
                     --output-dir data/DDR/preprocessed --no-border-removal
```

The options `--no-border-removal`, `--no-contrast` and `--alpha` produce the
configurations of the preprocessing ablation and of the grid search on alpha.

### 2. Data split

```bash
python make_splits.py --dataset aptos --labels data/aptos2019/train.csv \
                      --output splits/aptos2019_split.csv
python make_splits.py --dataset ddr --labels data/DDR/DR_grading \
                      --output splits/ddr_split.csv
```

APTOS 2019: stratified 85% / 15% hold-out, then 80% / 20% training / validation
inside the development partition (seed 42). The same split is used for both
tasks. DDR: stratified 70% / 15% / 15% of the gradable images.

### 3. Training

```bash
# Binary detection (No DR vs DR)
python train.py --task binary --splits splits/aptos2019_split.csv \
                --image-dir data/aptos2019/preprocessed --output-dir runs/binary \
                --feature-cache cache/aptos2019 --evaluate-test

# Five-class severity grading
python train.py --task multiclass --splits splits/aptos2019_split.csv \
                --image-dir data/aptos2019/preprocessed --output-dir runs/multiclass \
                --feature-cache cache/aptos2019 --evaluate-test
```

With frozen backbones, the embeddings are extracted once (batches of 64) and
cached, and only the projection layers and the head are trained (Adam,
learning rate 1e-4 with cosine decay to 1e-6, batch size 32 / 16, early stopping
with patience 10 on the validation loss, best epoch restored).

Each run writes `model.pt`, `config.json`, `history.csv`, `training_curves.png`
and, for the validation set (and the test set with `--evaluate-test`),
`metrics_*.json`, `report_*.txt`, `predictions_*.csv` and `confusion_matrix_*.png`.

The test set should be evaluated once, for the final configuration. All design
choices are made on the validation set.

### 4. Ablation studies (validation set)

| Study | Options |
|---|---|
| Backbone contribution | `--branches vit`, `--branches resnet`, `--branches vit resnet` |
| Fusion strategy | `--fusion concat`, `weighted_sum`, `product`, `bilinear` |
| Training regime | `--regime frozen`, `partial`, `full` |
| Class imbalance | `--loss ce`, `weighted_ce`, `focal`; `--oversample` |
| Preprocessing | train on folders produced with the preprocessing options above |

`partial` fine-tunes the last two encoder blocks of ViT and the last two
bottleneck blocks of ResNet50; `full` fine-tunes both backbones entirely.

### 5. Cross-validation

```bash
python cross_validate.py --task multiclass --splits splits/aptos2019_split.csv \
                         --image-dir data/aptos2019/preprocessed --output-dir runs/cv_multiclass \
                         --feature-cache cache/aptos2019
```

5 stratified folds of the development partition, repeated with seeds 42, 123
and 456 (15 runs). The test set is not used. The fold assignments are saved in
`fold_assignments.csv`.

### 6. Evaluation and statistical comparison

```bash
python evaluate.py --checkpoint runs/multiclass/model.pt --splits splits/aptos2019_split.csv \
                   --image-dir data/aptos2019/preprocessed --split test --output-dir runs/multiclass/eval

python compare_models.py \
    --comparison "5-class: fusion vs ResNet50" runs/multiclass/predictions_test.csv runs/multiclass_resnet/predictions_test.csv \
    --comparison "5-class: fusion vs ViT"      runs/multiclass/predictions_test.csv runs/multiclass_vit/predictions_test.csv \
    --output results/mcnemar.csv
```

Reported metrics: accuracy, balanced accuracy, per-class precision, recall,
F1 and specificity (one-vs-rest), macro and weighted F1, quadratic weighted
kappa, Expected Calibration Error (10 bins) and, for the binary task, ROC AUC,
sensitivity, specificity and PPV / NPV at thresholds 0.3, 0.5 and 0.7 for a
prevalence of 34%. Confidence intervals: Wilson score for proportions, delta
method for F1, Fisher z-transformation for QWK, DeLong for AUC. Paired
comparisons use McNemar's test (Edwards' continuity correction) with
Holm-Bonferroni adjustment.

### 7. Explainability

```bash
python explain.py --checkpoint runs/multiclass/model.pt --splits splits/aptos2019_split.csv \
                  --image-dir data/aptos2019/preprocessed --output-dir runs/multiclass/xai
```

By default one test image per grade is drawn at random and explained for the
predicted class; `--image-ids` selects specific images. Methods:

* **Grad-CAM** on the last bottleneck block of ResNet50 (`layer4[-1]`)
* **Integrated Gradients** from a black-image baseline, 50 steps
* **Saliency**, absolute input gradient
* **SmoothGrad**, n = 50, sigma = 0.15 x (max(x) - min(x))

### 8. Faithfulness of the attribution maps

```bash
python faithfulness.py --checkpoint runs/multiclass/model.pt --splits splits/aptos2019_split.csv \
                       --image-dir data/aptos2019/preprocessed --output-dir runs/multiclass/faithfulness
```

The top 10%, 20% and 30% most salient pixels (SmoothGrad) are replaced by the
mean pixel value of the image and the accuracy drop is compared with masking
the same number of random pixels (10 random masks per image).

## Reproducibility

* Seeds fixed to 42 for `random`, `numpy` and `torch`; deterministic cuDNN.
* Normalisation uses fixed ImageNet statistics, mean (0.485, 0.456, 0.406) and
  standard deviation (0.229, 0.224, 0.225), applied after preprocessing.
* Preprocessing depends only on the image being processed (its own mean).
* No data augmentation.
* Hyperparameters are listed in `drgrading/config.py`.

Trained weights are distributed as release assets rather than in the git
history (`*.pt` files are ignored by `.gitignore`).

## Intended use

This code is for research only. It is not a medical device, has not been
approved by any regulatory authority and must not be used for clinical
decisions.

## Citation

```bibtex
@article{eljattioui_vit_resnet_dr,
  title   = {A Hybrid Vision Transformer--ResNet50 Framework for Automated Detection
             and Severity Grading of Diabetic Retinopathy from Fundus Images},
  author  = {El Jattioui, Mohammed and Riffi, Jamal and Khodriss, Chaimae and
             Mahraz, Adnane Mohamed and Yahyaouy, Ali and Abdellaoui, Meriem and
             Benatiya Andaloussi, Idriss and Tairi, Hamid},
  journal = {(under review)},
  year    = {2026}
}
```
