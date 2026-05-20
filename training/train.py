#!/usr/bin/env python
"""
Phase 4 -- Model Training.

Fine-tunes YOLOv8s on the prepared chess-piece dataset.

Usage:
    python train.py [--epochs 50] [--batch 16] [--imgsz 640] [--resume]

Outputs:
    training/runs/chess_detector/  -- full YOLO training artefacts
    <project_root>/chess.pt        -- copy of the best weights for the app
"""

from __future__ import annotations

import argparse
import shutil

from ultralytics import YOLO

from config import (
    BATCH_SIZE,
    DATASET_YAML,
    EPOCHS,
    FINAL_WEIGHTS,
    IMGSZ,
    RUNS_DIR,
    WORKERS,
    YOLO_BASE_MODEL,
)

RUN_NAME = "chess_detector"


def train(
    epochs: int = EPOCHS,
    batch: int = BATCH_SIZE,
    imgsz: int = IMGSZ,
    resume: bool = False,
) -> None:
    model = YOLO(YOLO_BASE_MODEL)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        patience=10,
        save=True,
        save_period=10,
        resume=resume,
        verbose=True,
    )

    best_weights = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, FINAL_WEIGHTS)
        print(f"\nBest weights copied to {FINAL_WEIGHTS}")
    else:
        print("\nWARNING: best.pt not found -- check the training run output above.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 chess-piece detector.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--resume", action="store_true", help="Resume last run")
    args = parser.parse_args()

    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, resume=args.resume)


if __name__ == "__main__":
    main()
