"""Fundus image preprocessing.

Two operations are applied before feature extraction:

1. Black-border removal. The fundus camera produces a dark annulus around the
   retinal disc. The image is thresholded, the largest external contour (the
   retinal disc) is located and the image is cropped to its bounding box.

2. Mean-anchored contrast enhancement. Every pixel is stretched symmetrically
   around the mean intensity mu of the cropped image:

       I'(x, y) = mu + alpha * (I(x, y) - mu)

   which amplifies both bright lesions (exudates) and dark lesions
   (haemorrhages, microaneurysms). The result is clipped to [0, 255].

The image is finally resized to 224 x 224 pixels.
"""

from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .config import BORDER_THRESHOLD, CONTRAST_ALPHA, IMAGE_SIZE

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_rgb(image_path):
    """Read an image from disk and return it as an RGB uint8 array."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(image_rgb, output_path):
    """Write an RGB uint8 array to disk (PNG is lossless)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Cannot write image: {output_path}")


def remove_black_border(image_rgb, threshold=BORDER_THRESHOLD):
    """Crop the image to the bounding box of the retinal disc.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image, shape (H, W, 3), dtype uint8.
    threshold : int
        Grey level separating the black surround from the retina.

    Returns
    -------
    np.ndarray
        Cropped image. The input is returned unchanged when no contour is found.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # findContours returns 3 values in OpenCV 3 and 2 values in OpenCV 4;
    # the contour list is always the second-to-last element.
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return image_rgb

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image_rgb[y:y + h, x:x + w]


def enhance_contrast(image_rgb, alpha=CONTRAST_ALPHA):
    """Mean-anchored linear contrast stretch.

    The mean mu is computed over all pixels and channels of the image itself,
    so the transform depends only on the image being processed.
    """
    image = image_rgb.astype(np.float32)
    mu = image.mean()
    image = mu + alpha * (image - mu)
    return np.clip(image, 0, 255).astype(np.uint8)


def preprocess_image(image_rgb,
                     alpha=CONTRAST_ALPHA,
                     output_size=IMAGE_SIZE,
                     border_removal=True,
                     contrast=True,
                     threshold=BORDER_THRESHOLD):
    """Apply the full preprocessing chain to an RGB array.

    The two flags allow the four configurations of the preprocessing ablation
    (none, border removal only, contrast only, both).
    """
    if border_removal:
        image_rgb = remove_black_border(image_rgb, threshold=threshold)
    if contrast:
        image_rgb = enhance_contrast(image_rgb, alpha=alpha)
    return cv2.resize(image_rgb, (output_size, output_size),
                      interpolation=cv2.INTER_LINEAR)


def preprocess_fundus(image_path, alpha=CONTRAST_ALPHA, output_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Load an image from disk and apply border removal, contrast enhancement and resizing.

    Convenience wrapper reproducing the snippet given in the paper.

    Returns
    -------
    np.ndarray
        Preprocessed RGB image of shape (H, W, 3), dtype uint8.
    """
    image = load_rgb(image_path)
    return preprocess_image(image, alpha=alpha, output_size=output_size[0])


def _process_one(image_path, input_dir, output_dir, kwargs):
    """Worker used by preprocess_directory. Returns None on success, else an error."""
    try:
        relative = Path(image_path).relative_to(input_dir)
        output_path = Path(output_dir) / relative.with_suffix(".png")
        image = load_rgb(image_path)
        save_rgb(preprocess_image(image, **kwargs), output_path)
        return None
    except Exception as exc:  # keep processing the rest of the folder
        return f"{image_path}: {exc}"


def list_images(input_dir):
    """All image files found recursively in a directory, sorted."""
    input_dir = Path(input_dir)
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def preprocess_directory(input_dir, output_dir, workers=4, **kwargs):
    """Preprocess every image of a folder and write PNG files with the same names.

    Returns
    -------
    (int, list)
        Number of processed images and list of error messages.
    """
    input_dir = Path(input_dir)
    images = list_images(input_dir)
    worker = partial(_process_one, input_dir=input_dir, output_dir=output_dir, kwargs=kwargs)

    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(worker, images, chunksize=16)
    else:
        results = [worker(p) for p in images]

    errors = [r for r in results if r is not None]
    return len(images) - len(errors), errors
