#!/usr/bin/env python
"""
Phase 5 -- Evaluation.

Runs the trained model on the validation and test splits and prints key
detection metrics:

  * mAP50 / mAP50-95
  * Precision
  * Recall
  * Per-class AP
  * Confusion matrix  (saved as PNG)
  * Sample prediction visualisations

Usage:
    python validate.py [--weights ../chess.pt] [--split test] [--samples 16]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import CLASS_NAMES, DATASET_DIR, DATASET_YAML, FINAL_WEIGHTS, RUNS_DIR


def evaluate(
    weights: Path = FINAL_WEIGHTS,
    split: str = "test",
    n_samples: int = 16,
) -> None:
    if not weights.exists():
        print(f"Weights not found: {weights}")
        print("Train the model first with  python train.py")
        return

    model = YOLO(str(weights))

    # ---- metrics on the chosen split ----
    print(f"\n{'='*60}")
    print(f"  Evaluating on {split} split")
    print(f"{'='*60}\n")

    metrics = model.val(
        data=str(DATASET_YAML),
        split=split,
        project=str(RUNS_DIR),
        name=f"eval_{split}",
        exist_ok=True,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print("  Summary Metrics")
    print(f"{'='*60}")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        print(f"\n  Per-class AP50:")
        for i, ap in enumerate(metrics.box.ap50):
            cls_idx = metrics.box.ap_class_index[i] if i < len(metrics.box.ap_class_index) else i
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
            print(f"    {name:20s}  {ap:.4f}")

    # ---- confusion matrix ----
    cm_path = RUNS_DIR / f"eval_{split}" / "confusion_matrix.png"
    if cm_path.exists():
        print(f"\n  Confusion matrix saved to: {cm_path}")

    # ---- sample predictions ----
    _save_sample_predictions(model, split, n_samples)


def _save_sample_predictions(model: YOLO, split: str, n: int) -> None:
    """Run inference on *n* images from *split* and save annotated results."""
    img_dir = DATASET_DIR / "images" / split
    if not img_dir.exists():
        return

    out_dir = RUNS_DIR / f"eval_{split}" / "sample_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg"))[:n]
    if not images:
        images = sorted(img_dir.glob("*.png"))[:n]
    if not images:
        return

    for img_path in images:
        results = model(str(img_path), verbose=False)
        annotated = results[0].plot()
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

    print(f"  {len(images)} sample predictions saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained chess detector.")
    parser.add_argument("--weights", type=Path, default=FINAL_WEIGHTS)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--samples", type=int, default=16, help="Number of sample predictions")
    args = parser.parse_args()

    evaluate(weights=args.weights, split=args.split, n_samples=args.samples)


if __name__ == "__main__":
    main()
