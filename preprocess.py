"""Preprocess a folder of fundus images (border removal, contrast enhancement, resizing).

Examples
--------
APTOS 2019 (full pipeline):
    python preprocess.py --input-dir data/aptos2019/train_images \
                         --output-dir data/aptos2019/preprocessed

DDR (images are released without black borders, so border removal is skipped):
    python preprocess.py --input-dir data/DDR/DR_grading/images \
                         --output-dir data/DDR/preprocessed --no-border-removal

Preprocessing ablation (four configurations):
    python preprocess.py ... --no-border-removal --no-contrast   # raw (resize only)
    python preprocess.py ... --no-contrast                       # border removal only
    python preprocess.py ... --no-border-removal                 # contrast only
    python preprocess.py ...                                     # both (proposed)
"""

import argparse
import time

from drgrading.config import BORDER_THRESHOLD, CONTRAST_ALPHA, IMAGE_SIZE
from drgrading.preprocessing import preprocess_directory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Folder containing the raw images")
    parser.add_argument("--output-dir", required=True, help="Folder receiving the PNG outputs")
    parser.add_argument("--alpha", type=float, default=CONTRAST_ALPHA,
                        help="Contrast scaling factor (default: %(default)s)")
    parser.add_argument("--threshold", type=int, default=BORDER_THRESHOLD,
                        help="Grey level used to detect the black border (default: %(default)s)")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE,
                        help="Output side length in pixels (default: %(default)s)")
    parser.add_argument("--no-border-removal", action="store_true", help="Skip border removal")
    parser.add_argument("--no-contrast", action="store_true", help="Skip contrast enhancement")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    n_ok, errors = preprocess_directory(
        args.input_dir, args.output_dir, workers=args.workers,
        alpha=args.alpha, output_size=args.size, threshold=args.threshold,
        border_removal=not args.no_border_removal, contrast=not args.no_contrast)

    elapsed = time.time() - start
    print(f"Processed {n_ok} images in {elapsed:.1f} s -> {args.output_dir}")
    if n_ok:
        print(f"Average time per image: {1000 * elapsed / n_ok:.2f} ms (with {args.workers} workers)")
    for message in errors:
        print(f"Failed: {message}")


if __name__ == "__main__":
    main()
