#!/usr/bin/env python
"""
Phase 4 -- Inference Test.

Quick smoke-test that loads the trained ``chess.pt`` weights, runs inference on
one or more images, prints detected pieces, and optionally displays the result.

Usage:
    python inference_test.py image1.jpg [image2.jpg ...]
    python inference_test.py --test-split          # pick a random test image
    python inference_test.py --show image.jpg      # display in a window
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import CLASS_NAMES, DATASET_DIR, FINAL_WEIGHTS


def run_inference(image_paths: list[Path], weights: Path, show: bool = False) -> None:
    if not weights.exists():
        print(f"Model weights not found: {weights}")
        print("Train the model first with  python train.py")
        return

    model = YOLO(str(weights))

    for img_path in image_paths:
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        print(f"\n--- {img_path.name} ---")
        results = model(str(img_path), verbose=False)
        detections = results[0].boxes

        if detections is None or len(detections) == 0:
            print("  No pieces detected.")
            continue

        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  {name:20s}  conf={conf:.2f}  box=({x1},{y1})-({x2},{y2})")

        if show:
            annotated = results[0].plot()
            cv2.imshow(f"Inference: {img_path.name}", annotated)
            print("  Press any key to continue ...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with trained chess detector.")
    parser.add_argument("images", nargs="*", type=Path, help="Image path(s)")
    parser.add_argument("--weights", type=Path, default=FINAL_WEIGHTS)
    parser.add_argument("--show", action="store_true", help="Display annotated image")
    parser.add_argument(
        "--test-split",
        action="store_true",
        help="Pick a random image from the test split",
    )
    args = parser.parse_args()

    images: list[Path] = list(args.images)

    if args.test_split:
        test_dir = DATASET_DIR / "images" / "test"
        candidates = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if not candidates:
            print(f"No test images in {test_dir}")
            sys.exit(1)
        images.append(random.choice(candidates))

    if not images:
        parser.print_help()
        sys.exit(1)

    run_inference(images, weights=args.weights, show=args.show)


if __name__ == "__main__":
    main()
