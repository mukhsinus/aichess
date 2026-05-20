#!/usr/bin/env python
"""
Manual Dataset Detection & Normalization.

Detects a manually placed YOLO chess dataset inside ``training/dataset/`` and
prepares it for the training pipeline.  Two common download layouts are
supported and automatically normalised to the pipeline's Structure B format.

Structure A  (Roboflow-style -- per-split sub-folders):
    dataset/
    ├── train/  (or valid/ / val/ / test/)
    │   ├── images/
    │   └── labels/
    └── data.yaml

Structure B  (standard YOLO -- pipeline default):
    dataset/
    ├── images/{train, val, test}/
    ├── labels/{train, val, test}/
    └── dataset.yaml  (or data.yaml)

Usage:
    python detect_manual_dataset.py            # detect, normalize, report
    python detect_manual_dataset.py --no-fix   # report only, don't touch files
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("manual_dataset")

from config import (
    CLASS_NAMES,
    DATASET_DIR,
    DATASET_YAML,
    IMAGE_EXTENSIONS,
    NUM_CLASSES,
)

_SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
_LABEL_SKIP = {"classes.txt", "obj.names", "_classes.txt", "_classes_from_yaml.txt"}


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

def detect_structure(dataset_dir: Path) -> str | None:
    """Return ``'A'``, ``'B'``, or ``None`` depending on the layout found."""
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return None

    # Structure B: top-level images/ with split sub-dirs
    if (dataset_dir / "images").is_dir():
        for split in ("train", "val", "valid"):
            if (dataset_dir / "images" / split).is_dir():
                return "B"

    # Structure A: top-level train/ with images inside
    if (dataset_dir / "train").is_dir():
        train = dataset_dir / "train"
        if (train / "images").is_dir():
            return "A"
        if any(
            f.suffix.lower() in IMAGE_EXTENSIONS
            for f in train.iterdir()
            if f.is_file()
        ):
            return "A"

    return None


# ---------------------------------------------------------------------------
# Structure A → B normalisation
# ---------------------------------------------------------------------------

def _normalize_split_names(dataset_dir: Path) -> None:
    """Rename ``images/valid`` → ``images/val`` (same for labels) if needed."""
    for parent in ("images", "labels"):
        valid = dataset_dir / parent / "valid"
        val = dataset_dir / parent / "val"
        if valid.is_dir() and not val.is_dir():
            valid.rename(val)
            log.info("  Renamed %s/valid → %s/val", parent, parent)


def normalize_to_structure_b(dataset_dir: Path) -> bool:
    """Reorganise Structure A into Structure B **in place**.

    Returns True when ``images/train`` exists after conversion.
    """
    log.info("  Converting Structure A → Structure B ...")

    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"

    for src_name, dst_name in _SPLIT_ALIASES.items():
        split_dir = dataset_dir / src_name
        if not split_dir.is_dir():
            continue

        img_dst = images_root / dst_name
        lbl_dst = labels_root / dst_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if (split_dir / "images").is_dir():
            # Roboflow layout: images/ and labels/ inside each split
            for f in (split_dir / "images").iterdir():
                if f.is_file():
                    shutil.move(str(f), str(img_dst / f.name))
            if (split_dir / "labels").is_dir():
                for f in (split_dir / "labels").iterdir():
                    if f.is_file():
                        shutil.move(str(f), str(lbl_dst / f.name))
        else:
            # Flat layout: images and labels mixed in one directory
            for f in split_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    shutil.move(str(f), str(img_dst / f.name))
                elif f.suffix.lower() == ".txt" and f.name.lower() not in _LABEL_SKIP:
                    shutil.move(str(f), str(lbl_dst / f.name))

        try:
            shutil.rmtree(split_dir)
        except OSError:
            pass

    ok = (images_root / "train").is_dir()
    if ok:
        log.info("  Structure conversion complete")
    return ok


# ---------------------------------------------------------------------------
# YAML normalisation
# ---------------------------------------------------------------------------

def _find_bundled_yaml(dataset_dir: Path) -> Path | None:
    """Look for ``data.yaml`` or ``dataset.yaml`` inside the dataset dir."""
    for name in ("data.yaml", "dataset.yaml"):
        p = dataset_dir / name
        if p.exists():
            return p
    return None


def write_normalized_yaml(
    dataset_dir: Path,
    source_yaml: Path | None = None,
) -> None:
    """Write ``training/dataset.yaml`` with an **absolute** ``path:`` field.

    If *source_yaml* exists inside the dataset, its ``names`` field is
    preserved (after basic validation).  The absolute path prevents
    Ultralytics from resolving against its default ``datasets_dir`` setting.
    """
    names_dict: dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES)}
    nc = NUM_CLASSES

    if source_yaml is not None:
        try:
            import yaml

            src = yaml.safe_load(source_yaml.read_text())
            src_names = src.get("names")
            if src_names:
                if isinstance(src_names, list):
                    names_dict = {i: n for i, n in enumerate(src_names)}
                elif isinstance(src_names, dict):
                    names_dict = {int(k): v for k, v in src_names.items()}
                nc = int(src.get("nc", len(names_dict)))
        except Exception as exc:
            log.warning("  Could not parse %s: %s — using default class names", source_yaml, exc)

    abs_path = dataset_dir.resolve().as_posix()

    lines = [
        "# YOLOv8 dataset configuration -- Chess Piece Detection",
        "# Auto-generated for manual dataset mode.",
        "# Absolute path prevents Ultralytics from using its default datasets_dir.",
        "",
        f"path: {abs_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {nc}",
        "",
        "names:",
    ]
    for idx in sorted(names_dict.keys()):
        lines.append(f"  {idx}: {names_dict[idx]}")

    DATASET_YAML.write_text("\n".join(lines) + "\n")
    log.info("  Wrote %s  (path: %s)", DATASET_YAML.name, abs_path)


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_manual_dataset(dataset_dir: Path) -> dict:
    """Quick validation: image-label pairs, class coverage, augmentations.

    Returns a result dict consumed by :func:`check_manual_dataset`.
    """
    result: dict = {
        "valid": False,
        "total_images": 0,
        "total_labels": 0,
        "paired": 0,
        "class_ids_found": set(),
        "missing_classes": [],
        "has_augmentations": False,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.is_dir():
            result["splits"][split] = {"images": 0, "labels": 0, "paired": 0}
            continue

        images = {
            f.stem: f
            for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels: dict[str, Path] = {}
        if lbl_dir.is_dir():
            labels = {
                f.stem: f
                for f in lbl_dir.iterdir()
                if f.is_file()
                and f.suffix.lower() == ".txt"
                and f.name.lower() not in _LABEL_SKIP
            }

        paired = set(images) & set(labels)

        if any("_aug" in stem for stem in images):
            result["has_augmentations"] = True

        for stem in paired:
            try:
                for line in labels[stem].read_text().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        result["class_ids_found"].add(int(parts[0]))
            except Exception:
                pass

        result["total_images"] += len(images)
        result["total_labels"] += len(labels)
        result["paired"] += len(paired)
        result["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "paired": len(paired),
        }

    result["missing_classes"] = sorted(
        set(range(NUM_CLASSES)) - result["class_ids_found"]
    )
    result["valid"] = result["paired"] > 0
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_manual_dataset(fix: bool = True) -> dict:
    """Detect, validate, and optionally normalise a manual dataset.

    Returns a status dict with keys:
      - ``detected``      -- bool
      - ``structure``      -- ``'A'`` / ``'B'`` / ``None``
      - ``valid``          -- bool  (image-label pairs > 0)
      - ``skip_collect``   -- bool
      - ``skip_prepare``   -- bool
      - ``skip_augment``   -- bool  (True when augmentations already present)
      - ``validation``     -- full validation result dict
    """
    status: dict = {
        "detected": False,
        "structure": None,
        "valid": False,
        "skip_collect": False,
        "skip_prepare": False,
        "skip_augment": False,
        "validation": None,
    }

    structure = detect_structure(DATASET_DIR)
    if structure is None:
        return status

    log.info("")
    log.info("=" * 65)
    log.info("  MANUAL DATASET DETECTED  (Structure %s)", structure)
    log.info("=" * 65)

    status["detected"] = True
    status["structure"] = structure

    # Convert Structure A → B if needed
    if structure == "A" and fix:
        if not normalize_to_structure_b(DATASET_DIR):
            log.error("  Failed to normalise Structure A → B")
            return status

    # Rename valid/ → val/ if needed
    if fix:
        _normalize_split_names(DATASET_DIR)
        bundled = _find_bundled_yaml(DATASET_DIR)
        write_normalized_yaml(DATASET_DIR, source_yaml=bundled)

    # Validate the (now-normalised) dataset
    validation = validate_manual_dataset(DATASET_DIR)
    status["validation"] = validation
    status["valid"] = validation["valid"]

    log.info("  Total images:  %d", validation["total_images"])
    log.info("  Total labels:  %d", validation["total_labels"])
    log.info("  Paired:        %d", validation["paired"])
    for split, info in validation["splits"].items():
        log.info(
            "    %-5s  %d img, %d lbl, %d paired",
            split, info["images"], info["labels"], info["paired"],
        )

    if validation["missing_classes"]:
        names = [CLASS_NAMES[i] for i in validation["missing_classes"] if i < NUM_CLASSES]
        log.warning("  Missing class IDs: %s", validation["missing_classes"])
        log.warning("  Missing classes:   %s", names)
    else:
        log.info("  All %d chess piece classes represented", NUM_CLASSES)

    if validation["has_augmentations"]:
        log.info("  Augmented images already present in dataset")

    if validation["valid"]:
        status["skip_collect"] = True
        status["skip_prepare"] = True
        status["skip_augment"] = validation["has_augmentations"]
        log.info("  Dataset is READY for training")
    else:
        log.error("  Dataset NOT USABLE (no image-label pairs found)")

    log.info("=" * 65)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and validate a manually placed chess dataset.",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Report only — do not normalise or rewrite files",
    )
    args = parser.parse_args()

    result = check_manual_dataset(fix=not args.no_fix)

    if not result["detected"]:
        print(f"\nNo manual dataset found in {DATASET_DIR}")
        print("Place a YOLO chess dataset there and run again.")
        sys.exit(1)

    if result["valid"]:
        print("\nManual dataset is valid and ready for training.")
        if result["skip_augment"]:
            print("Augmentations already present — augmentation phase will be skipped.")
        sys.exit(0)
    else:
        print("\nManual dataset detected but not usable. Check the report above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
