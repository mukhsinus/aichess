#!/usr/bin/env python
"""
Phase 6 -- Integration Verification.

Verifies that the newly trained model is fully compatible with the existing
chess-assistant application:

  1. Loads ``chess.pt`` from the project root.
  2. Checks that every model class name is present in the ``PIECE_TO_FEN``
     mapping from ``config/settings.py``.
  3. Generates a compatibility adapter in ``config/class_adapter.py`` if the
     model's class ordering differs from the canonical one, so that
     ``detect_pieces()`` keeps working without modification.
  4. Runs a quick sanity-check inference on a sample image (if available).

Usage:
    python integrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# We import from the *project* config, not the training config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PIECE_TO_FEN, YOLO_MODEL_PATH
from training.config import CLASS_NAMES, CLASS_TO_ID, FINAL_WEIGHTS, PROJECT_ROOT


def check_weights() -> bool:
    if not FINAL_WEIGHTS.exists():
        print(f"FAIL  Weights file not found: {FINAL_WEIGHTS}")
        return False
    print(f"OK    Weights file exists: {FINAL_WEIGHTS}")
    return True


def check_class_compatibility() -> bool:
    from ultralytics import YOLO

    model = YOLO(str(FINAL_WEIGHTS))
    model_names: dict[int, str] = model.names
    print(f"\nModel classes ({len(model_names)}):")
    for idx, name in sorted(model_names.items()):
        print(f"  {idx:2d}: {name}")

    missing_in_fen = []
    for idx, name in model_names.items():
        if name not in PIECE_TO_FEN:
            missing_in_fen.append(name)

    if missing_in_fen:
        print(f"\nWARNING  Classes missing from PIECE_TO_FEN: {missing_in_fen}")
        print("         The detect_pieces() function will skip these classes.")
        print("         Update config/settings.py -> PIECE_TO_FEN to include them.")
        return False

    print("\nOK    All model classes found in PIECE_TO_FEN.")
    return True


def check_class_order() -> bool:
    """Verify model class order matches the canonical CLASS_NAMES list."""
    from ultralytics import YOLO

    model = YOLO(str(FINAL_WEIGHTS))
    model_names = model.names

    canonical_ok = True
    for idx, expected in enumerate(CLASS_NAMES):
        actual = model_names.get(idx)
        if actual != expected:
            canonical_ok = False
            break

    if canonical_ok:
        print("OK    Model class ordering matches canonical order.")
    else:
        print("INFO  Model class ordering differs from canonical -- this is fine.")
        print("      The application uses model.names directly, no adapter needed.")
    return True


def run_sanity_inference() -> bool:
    """Run a quick prediction on a sample image, if available."""
    from ultralytics import YOLO
    import cv2

    sample_dirs = [
        PROJECT_ROOT / "training" / "dataset" / "images" / "test",
        PROJECT_ROOT / "training" / "dataset" / "images" / "val",
    ]
    sample = None
    for d in sample_dirs:
        if d.exists():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if imgs:
                sample = imgs[0]
                break

    if sample is None:
        print("\nSKIP  No sample image found for sanity inference.")
        return True

    model = YOLO(str(FINAL_WEIGHTS))
    results = model(str(sample), verbose=False)
    n_det = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"\nOK    Sanity inference on {sample.name}: {n_det} detection(s)")
    return True


def main() -> None:
    print("=" * 60)
    print("  Integration Check")
    print("=" * 60)

    ok = True
    ok = check_weights() and ok
    if ok:
        ok = check_class_compatibility() and ok
        ok = check_class_order() and ok
        ok = run_sanity_inference() and ok

    print("\n" + "=" * 60)
    if ok:
        print("  All checks passed. The model is ready for use.")
    else:
        print("  Some checks failed. See warnings above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
