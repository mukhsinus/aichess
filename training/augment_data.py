#!/usr/bin/env python
"""
Phase 3 -- Data Augmentation.

Reads the prepared dataset from ``training/dataset/images/train`` and creates
augmented copies alongside the originals.  Only the **train** split is
augmented; validation and test remain untouched.

Augmentations applied (randomly composed per image):
  * Rotation (up to +-15 deg)
  * Perspective / affine distortion
  * Brightness & contrast changes
  * Shadow simulation (random darkening of one half)
  * Gaussian blur
  * Gaussian noise
  * Partial occlusion (random erasing / cutout)

Bounding boxes are transformed together with the image so labels stay correct.

Usage:
    python augment_data.py [--copies 3]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from config import AUG_PER_IMAGE, DATASET_DIR, IMG_SIZE


# ---------------------------------------------------------------------------
# Augmentation pipeline  (bbox_params keeps YOLO labels in sync)
# ---------------------------------------------------------------------------

def _build_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.Affine(shear=(-10, 10), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(int(IMG_SIZE * 0.05), int(IMG_SIZE * 0.12)),
                hole_width_range=(int(IMG_SIZE * 0.05), int(IMG_SIZE * 0.12)),
                fill=0,
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.3,
        ),
    )


# ---------------------------------------------------------------------------
# YOLO label I/O
# ---------------------------------------------------------------------------

def _read_yolo_labels(path: Path) -> tuple[list[list[float]], list[int]]:
    bboxes: list[list[float]] = []
    class_ids: list[int] = []
    if not path.exists():
        return bboxes, class_ids
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_ids.append(int(parts[0]))
        bboxes.append([float(x) for x in parts[1:5]])
    return bboxes, class_ids


def _write_yolo_labels(path: Path, bboxes: list, class_ids: list) -> None:
    lines = []
    for cls_id, bbox in zip(class_ids, bboxes):
        coords = " ".join(f"{v:.6f}" for v in bbox)
        lines.append(f"{cls_id} {coords}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main augmentation loop
# ---------------------------------------------------------------------------

def augment_train_split(copies: int = AUG_PER_IMAGE) -> None:
    img_dir = DATASET_DIR / "images" / "train"
    lbl_dir = DATASET_DIR / "labels" / "train"

    if not img_dir.exists():
        print(f"Train image directory not found: {img_dir}")
        print("Run prepare_dataset.py first.")
        return

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        print("No training images found.")
        return

    pipeline = _build_pipeline()
    created = 0

    for img_path in images:
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        bboxes, class_ids = _read_yolo_labels(lbl_path)
        if not bboxes:
            continue

        for c in range(copies):
            try:
                result = pipeline(
                    image=img,
                    bboxes=bboxes,
                    class_ids=class_ids,
                )
            except Exception:
                continue

            aug_img = result["image"]
            aug_bboxes = result["bboxes"]
            aug_ids = result["class_ids"]

            if not aug_bboxes:
                continue

            stem = img_path.stem
            out_img = img_dir / f"{stem}_aug{c}.jpg"
            out_lbl = lbl_dir / f"{stem}_aug{c}.txt"
            cv2.imwrite(str(out_img), aug_img)
            _write_yolo_labels(out_lbl, aug_bboxes, aug_ids)
            created += 1

    print(f"Augmentation complete: {created} augmented samples added to train split.")
    print(f"Total train images now: {len(list(img_dir.glob('*.jpg')))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment the training split.")
    parser.add_argument(
        "--copies",
        type=int,
        default=AUG_PER_IMAGE,
        help=f"Augmented copies per image (default: {AUG_PER_IMAGE})",
    )
    args = parser.parse_args()
    augment_train_split(copies=args.copies)


if __name__ == "__main__":
    main()
