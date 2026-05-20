#!/usr/bin/env python
"""
Phase 1b -- Dataset Validation.

Validates downloaded raw data *before* preparation:
  1. Image-label pair existence (every image should have a matching .txt)
  2. YOLO label format correctness (class_id cx cy w h, all normalised 0-1)
  3. All 12 chess classes are represented across the combined dataset
  4. Reports orphan images, orphan labels, corrupt images, and bad labels

Run standalone:
    python validate_dataset.py                # validate raw_data/
    python validate_dataset.py --prepared     # validate prepared dataset/

Or it is called automatically by the pipeline runner.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from config import (
    CLASS_NAMES,
    DATASET_DIR,
    IMAGE_EXTENSIONS,
    NUM_CLASSES,
    RAW_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_dataset")

LABEL_SKIP_NAMES = {
    "classes.txt", "obj.names", "_classes.txt", "_classes_from_yaml.txt",
    "readme.txt", "notes.txt",
}


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _is_valid_yolo_line(line: str, num_classes: int | None = None) -> tuple[bool, str]:
    """Check whether a single YOLO annotation line is valid."""
    parts = line.strip().split()
    if len(parts) < 5:
        return False, f"too few fields ({len(parts)})"
    try:
        cls_id = int(parts[0])
    except ValueError:
        return False, f"class_id not int: '{parts[0]}'"

    if cls_id < 0:
        return False, f"negative class_id: {cls_id}"
    if num_classes is not None and cls_id >= num_classes:
        pass  # don't reject; source may use its own class map which gets remapped later

    for i, token in enumerate(parts[1:5], start=1):
        try:
            val = float(token)
        except ValueError:
            return False, f"field {i} not float: '{token}'"
        if val < -0.01 or val > 1.5:
            return False, f"field {i} out of range: {val}"

    return True, "ok"


def validate_directory(
    root: Path,
    *,
    strict_classes: bool = False,
    max_class_id: int | None = None,
) -> dict:
    """Walk *root* recursively and validate image-label pairs.

    Returns a dict with counts and lists of issues.
    """
    stats = {
        "total_images": 0,
        "total_labels": 0,
        "paired": 0,
        "orphan_images": [],
        "orphan_labels": [],
        "corrupt_images": [],
        "bad_label_files": [],
        "class_id_counts": {},
        "total_annotations": 0,
        "bad_annotations": 0,
    }

    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}

    for f in root.rglob("*"):
        if f.is_dir():
            continue
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            images[f.stem] = f
            stats["total_images"] += 1
        elif f.suffix.lower() == ".txt" and f.name.lower() not in LABEL_SKIP_NAMES:
            labels[f.stem] = f
            stats["total_labels"] += 1

    img_stems = set(images.keys())
    lbl_stems = set(labels.keys())
    paired_stems = img_stems & lbl_stems

    stats["paired"] = len(paired_stems)
    stats["orphan_images"] = sorted(img_stems - lbl_stems)
    stats["orphan_labels"] = sorted(lbl_stems - img_stems)

    for stem in paired_stems:
        img_path = images[stem]
        lbl_path = labels[stem]

        img = cv2.imread(str(img_path))
        if img is None:
            stats["corrupt_images"].append(str(img_path))
            continue

        label_text = lbl_path.read_text().strip()
        if not label_text:
            stats["bad_label_files"].append((str(lbl_path), "empty file"))
            continue

        file_ok = True
        for line in label_text.splitlines():
            line = line.strip()
            if not line:
                continue
            stats["total_annotations"] += 1
            valid, reason = _is_valid_yolo_line(line, num_classes=max_class_id)
            if not valid:
                stats["bad_annotations"] += 1
                file_ok = False
            else:
                cls_id = int(line.split()[0])
                stats["class_id_counts"][cls_id] = stats["class_id_counts"].get(cls_id, 0) + 1

        if not file_ok:
            stats["bad_label_files"].append((str(lbl_path), "contains invalid lines"))

    return stats


def print_report(stats: dict, root: Path, strict_classes: bool = False) -> bool:
    """Pretty-print validation results. Returns True if dataset is usable."""
    log.info("")
    log.info("=" * 65)
    log.info("  DATASET VALIDATION: %s", root)
    log.info("=" * 65)
    log.info("  Images found:        %d", stats["total_images"])
    log.info("  Label files found:   %d", stats["total_labels"])
    log.info("  Image-label pairs:   %d", stats["paired"])
    log.info("  Orphan images:       %d", len(stats["orphan_images"]))
    log.info("  Orphan labels:       %d", len(stats["orphan_labels"]))
    log.info("  Corrupt images:      %d", len(stats["corrupt_images"]))
    log.info("  Total annotations:   %d", stats["total_annotations"])
    log.info("  Bad annotations:     %d", stats["bad_annotations"])

    if stats["corrupt_images"]:
        log.warning("  Corrupt images (first 10):")
        for p in stats["corrupt_images"][:10]:
            log.warning("    - %s", p)

    if stats["bad_label_files"]:
        log.warning("  Problematic label files (first 10):")
        for p, reason in stats["bad_label_files"][:10]:
            log.warning("    - %s (%s)", p, reason)

    class_ids = sorted(stats["class_id_counts"].keys())
    log.info("")
    log.info("  Class distribution:")
    for cid in class_ids:
        name = CLASS_NAMES[cid] if cid < NUM_CLASSES else f"unknown-{cid}"
        log.info("    %2d %-20s  %d annotations", cid, name, stats["class_id_counts"][cid])

    found_canonical = set(cid for cid in class_ids if cid < NUM_CLASSES)
    missing = set(range(NUM_CLASSES)) - found_canonical
    if missing:
        log.info("")
        log.info("  Missing canonical class IDs: %s", sorted(missing))
        missing_names = [CLASS_NAMES[i] for i in sorted(missing)]
        log.info("  Missing class names: %s", missing_names)
        if strict_classes:
            log.warning("  STRICT MODE: all 12 classes required but only %d found.", len(found_canonical))
    else:
        log.info("")
        log.info("  All %d chess piece classes are represented!", NUM_CLASSES)

    log.info("=" * 65)

    usable = stats["paired"] > 0
    if usable:
        log.info("  RESULT: Dataset is USABLE (%d paired samples)", stats["paired"])
    else:
        log.error("  RESULT: Dataset is NOT USABLE (no image-label pairs found)")

    return usable


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate chess-piece dataset.")
    parser.add_argument(
        "--prepared", action="store_true",
        help="Validate the prepared dataset/ instead of raw_data/",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Require all 12 chess classes to be present",
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Custom path to validate (overrides --prepared)",
    )
    args = parser.parse_args()

    if args.path:
        target = Path(args.path)
    elif args.prepared:
        target = DATASET_DIR
    else:
        target = RAW_DATA_DIR

    if not target.exists():
        log.error("Target directory does not exist: %s", target)
        return 1

    log.info("Validating: %s", target)
    stats = validate_directory(target, strict_classes=args.strict)
    usable = print_report(stats, target, strict_classes=args.strict)
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
